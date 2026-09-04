"""Three-way evidence boundaries for resident graph-capture full loops.

Warp CPU supplies the uncaptured resident baseline. Native CUDA capture is
optional and is skipped before qualification when the device or Warp capture
API is unavailable; it is never emulated on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution import Backend, Device
from particula.execution.communication import (
    CommunicationConfiguration,
    CommunicationMap,
    CommunicationMapForm,
    CommunicationResourceShape,
    CommunicationShapeKind,
    CommunicationTransportMode,
    PrescribedVolumeUpdate,
)
from particula.execution.graph_capture import (
    GraphCaptureAvailability,
    GraphCaptureCapability,
    GraphCaptureLifecycleState,
    GraphCaptureNativeCallables,
    ResidentGraphCaptureBinding,
    _attach_resident_graph_capture_binding,
    capture_prepared_resident_graph,
    close_resident_graph_capture,
    create_graph_capture_lifecycle,
    create_resident_graph_capture_signature,
    qualify_prepared_resident_graph_capture,
    replay_captured_resident_graph,
)
from particula.execution.process_adapters import ResidentNucleationRequest
from particula.execution.resident_scheduler import (
    enqueue_prepared_resident_simulation,
    prepare_resident_simulation,
)
from particula.execution.tests.full_loop_test import _build_loop_fixture
from particula.execution.tests.multi_box_loop_test import (
    _binding,
    _resident_graph,
    _scheduler_request,
)
from particula.gpu.kernels.nucleation import (
    NucleationConfig,
    NucleationExhaustionControls,
)
from particula.gpu.tests.cuda_availability import (
    CUDA_SKIP_REASON,
    cuda_available,
)
from particula.util.constants import GAS_CONSTANT

PARITY_RTOL = 1e-12
PARITY_ATOL = 1e-30


@dataclass
class _PreparedLoop:
    """Retain one authentic prepared resident loop and its resources."""

    wp: Any
    session: Any
    registry: Any
    guard: Any
    request: Any
    binding: ResidentGraphCaptureBinding
    prepared: Any
    coagulation_resources: Any
    wall_loss_resources: Any


class _WarpNativeCaptureAdapter:
    """Expose Warp's genuine native graph operations after the CUDA gate."""

    def __init__(self, wp: Any, native: str) -> None:
        self.wp = wp
        self.native = native

    def runtime_available(self) -> bool:
        """Return the already-established runtime qualification."""
        return True

    def device_available(self, device: Device) -> bool:
        """Require the exact pre-qualified native CUDA device."""
        return device.native == self.native

    def capture_api_available(self, device: Device) -> bool:
        """Require the exact device whose APIs were checked before setup."""
        return device.native == self.native

    def capture_callables(self, device: Device) -> GraphCaptureNativeCallables:
        """Bind Warp capture, replay, and exact-handle cleanup operations."""

        def capture_begin() -> None:
            self.wp.capture_begin(
                device=device.native,
                force_module_load=True,
            )

        def capture_release(handle: object) -> None:
            destroy = getattr(handle, "destroy", None)
            if callable(destroy):
                destroy()
            # Warp versions without a public destroy method release the native
            # graph when this exact opaque Python handle loses its final owner.

        return GraphCaptureNativeCallables(
            capture_begin,
            self.wp.capture_end,
            lambda: None,
            self.wp.capture_launch,
            capture_release,
        )


@pytest.fixture(autouse=True)
def _remove_resolver_schedule_registrations() -> Any:
    """Keep locally constructed schedules isolated between test cases."""
    import particula.execution.scheduler as scheduler_module

    schedules = scheduler_module._RESOLVER_SCHEDULES
    initial_schedules = list(schedules)
    _resident_graph.cache_clear()
    schedules.clear()
    yield
    _resident_graph.cache_clear()
    schedules[:] = initial_schedules


def _inventory_oracle(fixture: Any) -> np.ndarray:
    """Compute concentration-weighted inventory independently."""
    particles = fixture.session.particles
    gas = fixture.session.gas
    return particles.volume.numpy()[:, None] * (
        np.sum(
            particles.masses.numpy()
            * particles.concentration.numpy()[:, :, None],
            axis=1,
        )
        + gas.concentration.numpy()
    )


class _FakeNativeCapture:
    """Record native capture calls without emulating CUDA on Warp CPU."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.launches: list[object] = []
        self.handle = object()

    def callables(self) -> GraphCaptureNativeCallables:
        """Return the complete opaque native vocabulary used by P1--P3."""

        def capture_begin() -> None:
            self.calls.append("begin")

        def capture_end() -> object:
            self.calls.append("end")
            return self.handle

        def capture_instantiate() -> None:
            self.calls.append("instantiate")

        def capture_launch(handle: object) -> None:
            self.calls.append("launch")
            self.launches.append(handle)

        def capture_release(handle: object) -> None:
            self.calls.append("release")
            assert handle is self.handle

        return GraphCaptureNativeCallables(
            capture_begin,
            capture_end,
            capture_instantiate,
            capture_launch,
            capture_release,
        )

    def adapter(self) -> Any:
        """Return an adapter with deterministic ordered availability probes."""
        native = self.callables()
        owner = self

        class Adapter:
            """Expose the fake runtime and native callable vocabulary."""

            def runtime_available(self) -> bool:
                owner.calls.append("runtime")
                return True

            def device_available(self, device: Device) -> bool:
                owner.calls.append(f"device:{device.native}")
                return True

            def capture_api_available(self, device: Device) -> bool:
                owner.calls.append(f"api:{device.native}")
                return True

            def capture_callables(
                self, device: Device
            ) -> GraphCaptureNativeCallables:
                owner.calls.append(f"callables:{device.native}")
                return native

        return Adapter()


def _prepared_fake_capture(
    fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    duration: float,
) -> tuple[Any, Any, _FakeNativeCapture]:
    """Build an exact READY prepared binding and qualify it on fake CUDA."""
    import particula.execution.resident_scheduler as resident_scheduler

    request = fixture.request
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            request.session.metadata.device,
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)
    object.__setattr__(request.condensation, "time_step", duration)
    object.__setattr__(request.coagulation.request.state, "time_step", duration)
    object.__setattr__(request.dilution, "time_step", duration)
    object.__setattr__(request.wall_loss, "time_step", duration)
    object.__setattr__(request.nucleation, "time_step", duration)
    object.__setattr__(request.communication, "duration", duration)

    class PreparedAdapter:
        """Provide a write-free prepared operation for capture setup."""

        def prepare(self, _request: object) -> Any:
            return SimpleNamespace(execute=lambda: None)

    for name in (
        "WarpCondensationExecutionAdapter",
        "ResidentBrownianCoagulationExecutionAdapter",
        "ResidentDilutionAdapter",
        "ResidentWallLossAdapter",
        "ResidentNucleationAdapter",
    ):
        monkeypatch.setattr(resident_scheduler, name, PreparedAdapter)
    prepared = resident_scheduler.prepare_resident_simulation(request, duration)
    capture_set = fixture.registry.validate_capture_resource_set(
        request.capture_resource_requirements
    )

    cuda_device = Device(Backend.WARP, "cuda:0")
    object.__setattr__(request.session.metadata, "device", cuda_device)
    object.__setattr__(signature, "device", cuda_device)
    object.__setattr__(lifecycle.capability, "device", cuda_device)
    monkeypatch.setattr(
        request.registry, "validate_pinned_session", lambda _session: None
    )
    native = _FakeNativeCapture()
    qualification = qualify_prepared_resident_graph_capture(
        binding, prepared, capture_set, native.adapter()
    )
    return qualification, binding, native


def _snapshot(fixture: Any) -> tuple[np.ndarray, ...]:
    """Synchronize at an assertion boundary and copy meaningful state."""
    fixture.wp.synchronize()
    session = fixture.session
    return (
        session.particles.masses.numpy().copy(),
        session.particles.concentration.numpy().copy(),
        session.particles.charge.numpy().copy(),
        session.particles.density.numpy().copy(),
        session.particles.volume.numpy().copy(),
        session.gas.concentration.numpy().copy(),
        session.gas.vapor_pressure.numpy().copy(),
        session.gas.partitioning.numpy().copy(),
        session.environment.temperature.numpy().copy(),
        session.environment.pressure.numpy().copy(),
        session.environment.saturation_ratio.numpy().copy(),
        fixture.gas_snapshot.numpy().copy(),
        fixture.saturation_snapshot.numpy().copy(),
        fixture.request.coagulation.resources.rng_states.numpy().copy(),
        fixture.request.wall_loss.resources.rng_states.numpy().copy(),
    )


def _zero_nucleation_request(
    session: Any,
    registry: Any,
    resources: Any,
    duration: float,
) -> ResidentNucleationRequest:
    """Build a valid, deterministic no-admission nucleation request."""
    config = NucleationConfig(
        rate_law="activation",
        coefficient=0.0,
        survival_factor=1.0,
        precursor_index=0,
        molecule_counts=(1, 0),
        formation_diameter=1.0e-9,
        precursor_number_concentration_lower=0.0,
        precursor_number_concentration_upper=1.0e40,
        temperature_lower=100.0,
        temperature_upper=500.0,
    )
    return ResidentNucleationRequest(
        session,
        registry,
        resources,
        config,
        duration,
        NucleationExhaustionControls(True, False),
    )


def _capture_communication_configuration(session: Any, wp: Any) -> Any:
    """Build a closed, zero-rate one-dimensional map on the active device."""
    device = session.particles.masses.device
    edges = max(session.dimensions.n_boxes - 1, 0)
    sources = np.arange(edges, dtype=np.int32)
    destinations = sources + 1
    return CommunicationConfiguration(
        CommunicationMap(
            CommunicationMapForm.ONE_DIMENSIONAL,
            CommunicationTransportMode.GAS,
            edges,
            wp.array(sources, dtype=wp.int32, device=device),
            wp.array(destinations, dtype=wp.int32, device=device),
            wp.ones(edges, dtype=wp.int32, device=device),
            wp.zeros(edges, dtype=wp.float64, device=device),
        ),
        PrescribedVolumeUpdate(None),
        (
            CommunicationResourceShape(
                "edge_rates",
                wp.float64,
                CommunicationShapeKind.E,
            ),
        ),
    )


def _build_prepared_loop(
    device: str,
    n_boxes: int,
    duration: float,
    root_seed: int,
    *,
    selected_wall_loss_boxes: tuple[int, ...] = (),
) -> _PreparedLoop:
    """Build one real all-operation resident loop on the requested device."""
    manifest = tuple((f"box-{2 * index}", index) for index in range(n_boxes))
    session, registry, guard = _binding(device, manifest, root_seed)
    wp = pytest.importorskip("warp")
    _resident_graph.cache_clear()
    request, wall_loss, coagulation = _scheduler_request(
        session,
        registry,
        guard,
        duration,
        selected_wall_loss_boxes,
        _capture_communication_configuration(session, wp),
        session.dimensions.n_particles,
    )
    request = replace(
        request,
        nucleation=_zero_nucleation_request(
            session,
            registry,
            request.nucleation.resources,
            duration,
        ),
    )
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            session.metadata.device,
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request,
        session,
        registry,
        guard,
        lifecycle,
    )
    _attach_resident_graph_capture_binding(request, binding)
    prepared = prepare_resident_simulation(request, duration)
    return _PreparedLoop(
        wp=wp,
        session=session,
        registry=registry,
        guard=guard,
        request=request,
        binding=binding,
        prepared=prepared,
        coagulation_resources=coagulation,
        wall_loss_resources=wall_loss,
    )


def _prepared_snapshot(loop: _PreparedLoop) -> dict[str, np.ndarray]:
    """Synchronize once and copy all meaningful resident-loop outputs."""
    loop.wp.synchronize_device(loop.session.particles.masses.device)
    particles = loop.session.particles
    gas = loop.session.gas
    environment = loop.session.environment
    snapshot = {
        "particle_masses": particles.masses.numpy().copy(),
        "particle_concentration": particles.concentration.numpy().copy(),
        "particle_charge": particles.charge.numpy().copy(),
        "particle_density": particles.density.numpy().copy(),
        "particle_volume": particles.volume.numpy().copy(),
        "gas_molar_mass": gas.molar_mass.numpy().copy(),
        "gas_concentration": gas.concentration.numpy().copy(),
        "gas_vapor_pressure": gas.vapor_pressure.numpy().copy(),
        "gas_partitioning": gas.partitioning.numpy().copy(),
        "temperature": environment.temperature.numpy().copy(),
        "pressure": environment.pressure.numpy().copy(),
        "saturation_ratio": environment.saturation_ratio.numpy().copy(),
        "collision_pairs": (
            loop.coagulation_resources.collision_pairs.numpy().copy()
        ),
        "collision_counts": (
            loop.coagulation_resources.n_collisions.numpy().copy()
        ),
        "coagulation_rng": (
            loop.coagulation_resources.rng_states.numpy().copy()
        ),
        "wall_loss_rng": loop.wall_loss_resources.rng_states.numpy().copy(),
    }
    for registration in loop.request.diagnostics.registrations:
        snapshot[f"diagnostic_{registration.operation.value}"] = (
            registration.output.numpy().copy()
        )
    return snapshot


def _close_prepared_loop(loop: _PreparedLoop) -> None:
    """Release graph provenance before closing its resident session."""
    close_resident_graph_capture(loop.binding)
    loop.session.close(loop.registry, loop.guard)


def _assert_prepared_parity(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> None:
    """Compare physical outputs without requiring cross-device RNG words."""
    ignored = {"coagulation_rng", "wall_loss_rng"}
    assert actual.keys() == expected.keys()
    for name in actual.keys() - ignored:
        if actual[name].dtype.kind in {"b", "i", "u"}:
            npt.assert_equal(actual[name], expected[name], err_msg=name)
        else:
            npt.assert_allclose(
                actual[name],
                expected[name],
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
                err_msg=name,
            )


def _snapshot_inventory(snapshot: dict[str, np.ndarray]) -> np.ndarray:
    """Compute per-box/species inventory from a detached snapshot."""
    return snapshot["particle_volume"][:, None] * (
        np.sum(
            snapshot["particle_masses"]
            * snapshot["particle_concentration"][:, :, None],
            axis=1,
        )
        + snapshot["gas_concentration"]
    )


def _deterministic_numpy_oracle(
    initial: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Advance the deterministic fixture without reading production output."""
    expected = {name: value.copy() for name, value in initial.items()}
    expected_vapor_pressure = np.full_like(initial["gas_vapor_pressure"], 800.0)
    expected_saturation = (
        initial["gas_concentration"]
        * GAS_CONSTANT
        * initial["temperature"][:, None]
        / (initial["gas_molar_mass"][None, :] * expected_vapor_pressure)
    )
    expected_inventory = _snapshot_inventory(initial)
    expected["gas_vapor_pressure"] = expected_vapor_pressure
    expected["saturation_ratio"] = expected_saturation
    expected["diagnostic_gas_concentration_snapshot"] = initial[
        "gas_concentration"
    ].copy()
    expected["diagnostic_saturation_ratio_snapshot"] = (
        expected_saturation.copy()
    )
    expected["diagnostic_total_species_mass"] = expected_inventory
    expected["diagnostic_particle_number_concentration"] = np.sum(
        initial["particle_concentration"], axis=1
    )
    expected["diagnostic_latent_heat_energy"] = np.zeros_like(
        initial["gas_concentration"]
    )
    # The fixture's independently allocated baseline/source/sink ledgers are
    # zero, so the diagnostic definition reduces to total species inventory.
    expected["diagnostic_conservation_residual"] = expected_inventory.copy()
    return expected


def _independent_wall_loss_sink(
    initial: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
) -> np.ndarray:
    """Measure removed initial particle inventory without differencing totals."""
    removed = (initial["particle_concentration"] > 0.0) & (
        result["particle_concentration"] == 0.0
    )
    removed_mass = np.sum(
        initial["particle_masses"]
        * initial["particle_concentration"][:, :, None]
        * removed[:, :, None],
        axis=1,
    )
    return initial["particle_volume"][:, None] * removed_mass


def _wall_loss_removal_moments(
    snapshot: dict[str, np.ndarray],
    duration: float,
    steps: int,
) -> tuple[float, float]:
    """Return independent Bernoulli mean and variance for active slots."""
    from particula.dynamics.properties.wall_loss_coefficient import (
        get_spherical_wall_loss_coefficient_via_system_state,
    )

    mean = 0.0
    variance = 0.0
    density = snapshot["particle_density"]
    for box in range(snapshot["particle_masses"].shape[0]):
        for slot in range(snapshot["particle_masses"].shape[1]):
            concentration = snapshot["particle_concentration"][box, slot]
            masses = snapshot["particle_masses"][box, slot]
            if concentration <= 0.0 or np.sum(masses) <= 0.0:
                continue
            particle_volume = np.sum(masses / density)
            radius = (3.0 * particle_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
            effective_density = float(np.sum(masses) / particle_volume)
            coefficient = float(
                get_spherical_wall_loss_coefficient_via_system_state(
                    wall_eddy_diffusivity=1.0,
                    particle_radius=radius,
                    particle_density=effective_density,
                    temperature=float(snapshot["temperature"][box]),
                    pressure=float(snapshot["pressure"][box]),
                    chamber_radius=1.0,
                )
            )
            probability = 1.0 - np.exp(-coefficient * duration * steps)
            mean += probability
            variance += probability * (1.0 - probability)
    return mean, variance


def _assert_same_state(
    actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...]
) -> None:
    """Compare float state tightly and integer metadata exactly."""
    for index, (actual_value, expected_value) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if index in {7, 12, 13, 14}:
            npt.assert_equal(actual_value, expected_value)
        else:
            npt.assert_allclose(
                actual_value,
                expected_value,
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "communication_family",
    (CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES),
)
def test_deterministic_full_loop_matches_independent_uncaptured_baseline(
    monkeypatch: pytest.MonkeyPatch,
    communication_family: CommunicationTransportMode,
) -> None:
    """Three zero-duration steps preserve the full uncaptured resident state."""
    reference = _build_loop_fixture(monkeypatch, communication_family)
    initial_inventory = _inventory_oracle(reference)

    reference.scheduler.execute(0.0)
    stable_state = _snapshot(reference)
    for _ in range(2):
        reference.scheduler.execute(0.0)
        _assert_same_state(_snapshot(reference), stable_state)
        npt.assert_allclose(
            _inventory_oracle(reference),
            initial_inventory,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )

    expected_dispatch = tuple(
        node_id
        for node_id in reference.request.schedule.ordered_node_ids
        if node_id not in {"vapor_pressure_refresh", "saturation_refresh"}
    )
    assert reference.trace == list(expected_dispatch * 3)
    assert reference.guard.completed_steps == 3


def _require_native_cuda_capture() -> Any:
    """Skip only before native qualification when CUDA capture is unavailable."""
    wp = pytest.importorskip("warp")
    if not cuda_available(wp):
        pytest.skip(CUDA_SKIP_REASON)
    if not any(str(device).startswith("cuda") for device in wp.get_devices()):
        pytest.skip(CUDA_SKIP_REASON)
    if not all(
        callable(getattr(wp, name, None))
        for name in ("capture_begin", "capture_end", "capture_launch")
    ):
        pytest.skip("Warp capture API unavailable")
    return wp


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_real_uncaptured_warp_cpu_nonzero_loop_matches_numpy() -> None:
    """Exercise the native-test fixture on the required uncaptured baseline."""
    loop = _build_prepared_loop("cpu", 3, 0.25, 1571)
    try:
        initial = _prepared_snapshot(loop)
        expected = _deterministic_numpy_oracle(initial)
        for _ in range(3):
            enqueue_prepared_resident_simulation(loop.prepared)
        result = _prepared_snapshot(loop)

        _assert_prepared_parity(result, expected)
        npt.assert_allclose(
            _snapshot_inventory(result),
            _snapshot_inventory(initial),
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )
        assert loop.guard.completed_steps == 3
    finally:
        _close_prepared_loop(loop)


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.parametrize("n_boxes", (1, 3))
def test_native_cuda_nonzero_full_loop_matches_numpy_and_uncaptured_warp(
    n_boxes: int,
) -> None:
    """Replay a genuine nonzero CUDA graph against Warp CPU and NumPy."""
    wp = _require_native_cuda_capture()
    duration = 0.25
    steps = 3
    cpu_loop = _build_prepared_loop("cpu", n_boxes, duration, 1571)
    cuda_loop = None
    captured = None
    try:
        cuda_loop = _build_prepared_loop("cuda", n_boxes, duration, 1571)
        cpu_initial = _prepared_snapshot(cpu_loop)
        cuda_initial = _prepared_snapshot(cuda_loop)
        _assert_prepared_parity(cuda_initial, cpu_initial)
        expected = _deterministic_numpy_oracle(cpu_initial)
        initial_inventory = _snapshot_inventory(cpu_initial)

        capture_set = cuda_loop.registry.validate_capture_resource_set(
            cuda_loop.request.capture_resource_requirements
        )
        adapter = _WarpNativeCaptureAdapter(
            wp,
            cuda_loop.session.metadata.device.native,
        )
        qualification = qualify_prepared_resident_graph_capture(
            cuda_loop.binding,
            cuda_loop.prepared,
            capture_set,
            adapter,
        )
        captured = capture_prepared_resident_graph(qualification)
        # Native recording itself is not a hidden physical launch.
        _assert_prepared_parity(_prepared_snapshot(cuda_loop), cuda_initial)

        for _ in range(steps):
            enqueue_prepared_resident_simulation(cpu_loop.prepared)
            replay_captured_resident_graph(captured, qualification.duration)

        cpu_result = _prepared_snapshot(cpu_loop)
        cuda_result = _prepared_snapshot(cuda_loop)
        _assert_prepared_parity(cuda_result, cpu_result)
        _assert_prepared_parity(cpu_result, expected)
        _assert_prepared_parity(cuda_result, expected)
        npt.assert_allclose(
            _snapshot_inventory(cuda_result),
            initial_inventory,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )
        npt.assert_equal(cuda_result["gas_partitioning"], 0)
        assert cpu_loop.guard.completed_steps == steps
        assert cuda_loop.guard.completed_steps == steps
        assert not hasattr(captured, "handle")
    finally:
        captured = None
        if cuda_loop is not None:
            _close_prepared_loop(cuda_loop)
        _close_prepared_loop(cpu_loop)


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.stochastic
def test_native_cuda_rng_continuation_and_stochastic_aggregate() -> None:
    """Compare wall-loss aggregates without cross-device trajectories."""
    wp = _require_native_cuda_capture()
    duration = 1.0
    steps = 2
    seeds = range(12)
    cpu_removed = 0
    cuda_removed = 0
    expected_removed = 0.0
    expected_variance = 0.0
    continued_words = 0

    for seed in seeds:
        selected = (0, 1, 2)
        cpu_loop = _build_prepared_loop(
            "cpu",
            len(selected),
            duration,
            seed,
            selected_wall_loss_boxes=selected,
        )
        cuda_loop = None
        captured = None
        try:
            cuda_loop = _build_prepared_loop(
                "cuda",
                len(selected),
                duration,
                seed,
                selected_wall_loss_boxes=selected,
            )
            cpu_initial = _prepared_snapshot(cpu_loop)
            cuda_initial = _prepared_snapshot(cuda_loop)
            _assert_prepared_parity(cuda_initial, cpu_initial)
            mean, variance = _wall_loss_removal_moments(
                cpu_initial,
                duration,
                steps,
            )
            expected_removed += mean
            expected_variance += variance

            capture_set = cuda_loop.registry.validate_capture_resource_set(
                cuda_loop.request.capture_resource_requirements
            )
            qualification = qualify_prepared_resident_graph_capture(
                cuda_loop.binding,
                cuda_loop.prepared,
                capture_set,
                _WarpNativeCaptureAdapter(
                    wp,
                    cuda_loop.session.metadata.device.native,
                ),
            )
            captured = capture_prepared_resident_graph(qualification)
            cuda_rng_initial = cuda_initial["wall_loss_rng"]

            enqueue_prepared_resident_simulation(cpu_loop.prepared)
            replay_captured_resident_graph(captured, qualification.duration)
            cuda_first = _prepared_snapshot(cuda_loop)
            assert np.any(cuda_first["wall_loss_rng"] != cuda_rng_initial)

            enqueue_prepared_resident_simulation(cpu_loop.prepared)
            replay_captured_resident_graph(captured, qualification.duration)
            cpu_result = _prepared_snapshot(cpu_loop)
            cuda_result = _prepared_snapshot(cuda_loop)
            continued_words += int(
                np.count_nonzero(
                    cuda_result["wall_loss_rng"] != cuda_first["wall_loss_rng"]
                )
            )

            cpu_removed += int(
                np.count_nonzero(
                    (cpu_initial["particle_concentration"] > 0.0)
                    & (cpu_result["particle_concentration"] == 0.0)
                )
            )
            cuda_removed += int(
                np.count_nonzero(
                    (cuda_initial["particle_concentration"] > 0.0)
                    & (cuda_result["particle_concentration"] == 0.0)
                )
            )
            for initial, result in (
                (cpu_initial, cpu_result),
                (cuda_initial, cuda_result),
            ):
                initial_inventory = _snapshot_inventory(initial)
                final_inventory = _snapshot_inventory(result)
                sink_inventory = _independent_wall_loss_sink(initial, result)
                assert np.all(final_inventory <= initial_inventory)
                npt.assert_allclose(
                    final_inventory + sink_inventory,
                    initial_inventory,
                    rtol=PARITY_RTOL,
                    atol=PARITY_ATOL,
                )
        finally:
            captured = None
            if cuda_loop is not None:
                _close_prepared_loop(cuda_loop)
            _close_prepared_loop(cpu_loop)

    sigma = np.sqrt(expected_variance)
    single_path_bound = max(4.0 * sigma, 2.0)
    cross_path_bound = max(4.0 * np.sqrt(2.0) * sigma, 2.0)
    assert abs(cpu_removed - expected_removed) <= single_path_bound
    assert abs(cuda_removed - expected_removed) <= single_path_bound
    assert abs(cuda_removed - cpu_removed) <= cross_path_bound
    assert continued_words > 0


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_cuda_capture_support_is_native_only() -> None:
    """Native CUDA is an optional capture boundary, never a Warp-CPU fallback."""
    wp = _require_native_cuda_capture()
    devices = [
        device for device in wp.get_devices() if str(device).startswith("cuda")
    ]
    assert devices


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_cuda_capture_zero_duration_contract_is_available_before_capture() -> (
    None
):
    """CUDA availability is decided before a zero-duration capture is attempted."""
    wp = _require_native_cuda_capture()
    assert callable(wp.capture_begin)
    assert callable(wp.capture_end)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "communication_family",
    (CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES),
)
def test_fake_native_capture_replays_nonzero_reference_steps_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    communication_family: CommunicationTransportMode,
) -> None:
    """Capture and replay preserve a deterministic multi-step reference state."""
    fixture = _build_loop_fixture(monkeypatch, communication_family)
    qualification, binding, native = _prepared_fake_capture(
        fixture, monkeypatch, 1.25
    )
    import particula.execution.resident_scheduler as resident_scheduler

    monkeypatch.setattr(
        resident_scheduler,
        "_enqueue_captured_prepared_operations",
        lambda _prepared: None,
    )

    try:
        before = _snapshot(fixture)
        captured = capture_prepared_resident_graph(qualification)
        after_capture = _snapshot(fixture)
        replay_captured_resident_graph(captured, qualification.duration)
        after_replay = _snapshot(fixture)

        _assert_same_state(after_capture, before)
        _assert_same_state(after_replay, before)
        replay_captured_resident_graph(captured, qualification.duration)
        _assert_same_state(_snapshot(fixture), before)
        assert not hasattr(captured, "handle")
        assert native.calls == [
            "runtime",
            "device:cuda:0",
            "api:cuda:0",
            "callables:cuda:0",
            "begin",
            "end",
            "launch",
            "launch",
        ]
        assert native.launches == [native.handle, native.handle]
        assert binding.lifecycle.state is GraphCaptureLifecycleState.CAPTURED
        assert fixture.guard.completed_steps == 2
    finally:
        close_resident_graph_capture(binding)
    assert native.calls[-1] == "release"
    assert native.calls.count("release") == 1


@pytest.mark.warp
def test_captured_replay_rejects_stale_binding_and_duration_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duration and terminal binding drift are rejected without native launch."""
    fixture = _build_loop_fixture(monkeypatch, CommunicationTransportMode.GAS)
    qualification, binding, native = _prepared_fake_capture(
        fixture, monkeypatch, 2.0
    )
    import particula.execution.resident_scheduler as resident_scheduler

    monkeypatch.setattr(
        resident_scheduler,
        "_enqueue_captured_prepared_operations",
        lambda _prepared: None,
    )
    captured = capture_prepared_resident_graph(qualification)

    try:
        with pytest.raises(ValueError, match="duration"):
            replay_captured_resident_graph(captured, 1.0)
        assert native.launches == []
        assert fixture.guard.completed_steps == 0
    finally:
        close_resident_graph_capture(binding)
    with pytest.raises(ValueError, match="not P2-issued"):
        replay_captured_resident_graph(captured, qualification.duration)
    assert native.launches == []
    assert native.calls.count("release") == 1


@pytest.mark.warp
def test_capture_replay_has_no_forbidden_host_work_and_preserves_rng_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake native capture proves no transfer, sync, instantiate, or RNG reset."""
    fixture = _build_loop_fixture(monkeypatch, CommunicationTransportMode.GAS)
    qualification, binding, native = _prepared_fake_capture(
        fixture, monkeypatch, 0.0
    )
    import particula.execution.resident_scheduler as resident_scheduler
    from particula.gpu import conversion

    monkeypatch.setattr(
        resident_scheduler,
        "_enqueue_captured_prepared_operations",
        lambda _prepared: None,
    )
    monkeypatch.setattr(
        fixture.wp,
        "synchronize",
        lambda: pytest.fail("capture/replay must not synchronize"),
    )
    monkeypatch.setattr(
        conversion,
        "to_warp_particle_data",
        lambda *_args, **_kwargs: pytest.fail(
            "capture/replay must not transfer"
        ),
    )
    monkeypatch.setattr(
        conversion,
        "to_warp_gas_data",
        lambda *_args, **_kwargs: pytest.fail(
            "capture/replay must not transfer"
        ),
    )
    monkeypatch.setattr(
        conversion,
        "to_warp_environment_data",
        lambda *_args, **_kwargs: pytest.fail(
            "capture/replay must not transfer"
        ),
    )

    coagulation_rng = (
        fixture.request.coagulation.resources.rng_states.numpy().copy()
    )
    wall_loss_rng = (
        fixture.request.wall_loss.resources.rng_states.numpy().copy()
    )
    captured = capture_prepared_resident_graph(qualification)
    try:
        replay_captured_resident_graph(captured, qualification.duration)

        assert "instantiate" not in native.calls
        npt.assert_equal(
            fixture.request.coagulation.resources.rng_states.numpy(),
            coagulation_rng,
        )
        npt.assert_equal(
            fixture.request.wall_loss.resources.rng_states.numpy(),
            wall_loss_rng,
        )
        assert fixture.guard.completed_steps == 1
    finally:
        close_resident_graph_capture(binding)
