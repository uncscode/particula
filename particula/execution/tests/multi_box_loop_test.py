"""Multi-box resident lifecycle, logical-ID, and wall-loss regressions.

The rows deliberately use complete CPU carriers and the real resident resource
boundary.  Snapshots synchronize only at assertion boundaries; no test retains
a resident binding after its case completes.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from functools import cache
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution import (
    CONDENSATION_CAPABILITY_MATRIX,
    CONDENSATION_PROCESS,
    Backend,
    CapabilityRequirements,
    CondensationActivityMode,
    CondensationConfiguration,
    CondensationExecutionMode,
    CondensationSurfaceMode,
    Device,
)
from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig,
    ResidentBrownianCoagulationExecutionState,
    WarpBrownianCoagulationExecutionState,
    WarpBrownianCoagulationState,
)
from particula.execution.adapters.condensation import (
    CondensationExecutionConfig,
    WarpCondensationExecutionState,
    WarpCondensationState,
)
from particula.execution.communication import (
    CommunicationConfiguration,
    CommunicationMap,
    CommunicationMapForm,
    CommunicationResourceShape,
    CommunicationShapeKind,
    CommunicationTransportMode,
    PrescribedVolumeUpdate,
)
from particula.execution.diagnostics import (
    ResidentDiagnosticOperation,
    ResidentDiagnosticRegistration,
    ResidentDiagnosticsPlan,
)
from particula.execution.gpu_resources import GPUResourceRegistry
from particula.execution.gpu_session import (
    ResidentStepGuard,
    setup_resident_session,
)
from particula.execution.process_adapters import (
    ResidentDilutionRequest,
    ResidentNucleationRequest,
    ResidentWallLossRequest,
)
from particula.execution.process_graph import (
    DependencyEdge,
    ProcessNode,
    TimestepPlan,
)
from particula.execution.resident_communication import (
    ResidentCommunicationRequest,
)
from particula.execution.resident_scheduler import (
    ResidentSimulationRequest,
    ResidentSimulationScheduler,
)
from particula.execution.scheduler import (
    EnabledNodeSelection,
    NucleationCondensationDirection,
    SchedulerProfile,
    resolve_timestep_schedule,
)
from particula.execution.state_updates import (
    ResidentEnvironmentUpdateRequest,
    ResidentGasUpdateRequest,
)
from particula.gas import EnvironmentData, GasData
from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
from particula.gpu.tests.cuda_availability import CUDA_SKIP_REASON, warp_devices
from particula.particles import ParticleData

PARITY_RTOL = 1e-12
PARITY_ATOL = 1e-30
INVENTORY_RTOL = 1e-12
INVENTORY_ATOL = 1e-30

_NODE_IDS = (
    "communication",
    "volume_evolution",
    "environment_update",
    "gas_update",
    "vapor_pressure_refresh",
    "saturation_refresh",
    "condensation",
    "brownian_coagulation",
    "dilution",
    "wall_loss",
    "nucleation",
    "diagnostics",
)


@pytest.fixture(autouse=True)
def _isolate_wall_loss_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep scheduler dispatch real while isolating wall-loss behavior."""
    import particula.execution.resident_scheduler as resident_scheduler

    class NoOpAdapter:
        """Accept one unrelated process dispatch without mutating state."""

        def execute(self, _request: object) -> None:
            pass

    for name in (
        "WarpCondensationExecutionAdapter",
        "ResidentBrownianCoagulationExecutionAdapter",
        "ResidentDilutionAdapter",
        "ResidentNucleationAdapter",
    ):
        monkeypatch.setattr(resident_scheduler, name, NoOpAdapter)


def _require_device(device: str) -> Any:
    """Lazily require Warp and skip an unavailable optional CUDA device."""
    wp = pytest.importorskip("warp")
    if device not in warp_devices(wp):
        pytest.skip(CUDA_SKIP_REASON)
    return wp


def _cpu_carriers(
    manifest: tuple[tuple[str, int], ...],
) -> tuple[ParticleData, GasData, EnvironmentData]:
    """Build independent 16-slot, two-species rows at manifest lanes."""
    n_boxes = len(manifest)
    masses = np.zeros((n_boxes, 16, 2), dtype=np.float64)
    concentration = np.zeros((n_boxes, 16), dtype=np.float64)
    charge = np.zeros((n_boxes, 16), dtype=np.float64)
    gas_concentration = np.zeros((n_boxes, 2), dtype=np.float64)
    temperature = np.zeros(n_boxes, dtype=np.float64)
    pressure = np.zeros(n_boxes, dtype=np.float64)
    volume = np.zeros(n_boxes, dtype=np.float64)
    for logical_id, lane in manifest:
        # ``box-3`` and ``free`` are valid no-work rows in every arrangement.
        active_slots = (
            0
            if logical_id in {"box-3", "free"}
            else 1 + (sum(ord(character) for character in logical_id) % 2)
        )
        value = float(sum(ord(character) for character in logical_id))
        masses[lane, :active_slots] = np.array(
            [1.0e-20 * value, 2.0e-20 * value],
            dtype=np.float64,
        )
        concentration[lane, :active_slots] = 1.0 + value / 1000.0
        gas_concentration[lane] = np.array(
            [1.0e-11 * value, 2.0e-11 * value],
            dtype=np.float64,
        )
        temperature[lane] = 295.0 + value / 100.0
        pressure[lane] = 101325.0 - value
        volume[lane] = 1.0e-8 * value
    particles = ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=np.array([1000.0, 1200.0], dtype=np.float64),
        volume=volume,
    )
    gas = GasData(
        name=["species-a", "species-b"],
        molar_mass=np.array([0.018, 0.098], dtype=np.float64),
        concentration=gas_concentration,
        partitioning=np.array([False, False]),
    )
    environment = EnvironmentData(
        temperature=temperature,
        pressure=pressure,
        saturation_ratio=np.ones((n_boxes, 2), dtype=np.float64),
    )
    assert particles.masses.shape == (n_boxes, 16, 2)
    assert particles.concentration.shape == (n_boxes, 16)
    assert gas.concentration.shape == (n_boxes, 2)
    return particles, gas, environment


def _binding(
    device: str,
    manifest: tuple[tuple[str, int], ...],
    root_seed: int = 1531,
) -> tuple[Any, GPUResourceRegistry, ResidentStepGuard]:
    """Upload one manifest fixture and bind its exact registry and guard."""
    wp = _require_device(device)
    particles, gas, environment = _cpu_carriers(manifest)
    ids, lanes = zip(*manifest, strict=True)
    session = setup_resident_session(
        particles,
        gas,
        environment,
        Device(Backend.WARP, str(wp.get_device(device))),
        root_seed=root_seed,
        logical_box_ids=ids,
        lanes=lanes,
    )
    registry = GPUResourceRegistry(session)
    return session, registry, ResidentStepGuard(session, registry)


@pytest.fixture
def resident_factory() -> Generator[Any, None, None]:
    """Close each created binding in reverse order, including failed cases."""
    bindings: list[tuple[Any, GPUResourceRegistry, ResidentStepGuard]] = []

    def create(
        *args: Any, **kwargs: Any
    ) -> tuple[Any, GPUResourceRegistry, ResidentStepGuard]:
        session, registry, guard = _binding(*args, **kwargs)
        bindings.append((session, registry, guard))
        return session, registry, guard

    try:
        yield create
    except BaseException:
        _close_bindings(bindings, raise_errors=False)
        raise
    else:
        _close_bindings(bindings)


def _close_bindings(
    bindings: list[tuple[Any, Any, Any]], *, raise_errors: bool = True
) -> None:
    """Attempt every binding close and then surface the first cleanup error."""
    cleanup_errors: list[BaseException] = []
    for session, registry, guard in reversed(bindings):
        try:
            session.close(registry, guard)
        except BaseException as error:  # pragma: no cover - defensive teardown
            cleanup_errors.append(error)
    if cleanup_errors and raise_errors:
        raise cleanup_errors[0]


def test_binding_cleanup_continues_after_close_failure() -> None:
    """A failed close does not prevent cleanup of older resident bindings."""
    calls: list[str] = []

    class Session:
        """Record cleanup order and optionally reject close."""

        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self, _registry: object, _guard: object) -> None:
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} close failed")

    bindings = [
        (Session("older"), object(), object()),
        (Session("newer", fail=True), object(), object()),
    ]

    with pytest.raises(RuntimeError, match="newer close failed"):
        _close_bindings(bindings)

    assert calls == ["newer", "older"]


@contextmanager
def _temporary_binding(
    device: str,
    manifest: tuple[tuple[str, int], ...],
    root_seed: int,
) -> Generator[tuple[Any, GPUResourceRegistry, ResidentStepGuard], None, None]:
    """Yield one binding and release it before the next aggregate iteration."""
    session, registry, guard = _binding(device, manifest, root_seed)
    try:
        yield session, registry, guard
    except BaseException:
        try:
            session.close(registry, guard)
        except BaseException as cleanup_error:
            _ = cleanup_error
        raise
    else:
        session.close(registry, guard)


def _lane(session: Any, logical_id: str) -> int:
    """Resolve a physical row through immutable logical-ID stream metadata."""
    stream = session.metadata.stream
    return stream.lanes[stream.logical_box_ids.index(logical_id)]


def _snapshot(session: Any) -> tuple[np.ndarray, ...]:
    """Synchronize once and copy mutable resident state for comparison."""
    wp = pytest.importorskip("warp")
    wp.synchronize_device(session.particles.masses.device)
    return (
        session.particles.masses.numpy().copy(),
        session.particles.concentration.numpy().copy(),
        session.particles.charge.numpy().copy(),
        session.gas.concentration.numpy().copy(),
        session.gas.vapor_pressure.numpy().copy(),
        session.environment.temperature.numpy().copy(),
        session.environment.pressure.numpy().copy(),
        session.environment.saturation_ratio.numpy().copy(),
    )


def _primary_snapshot(session: Any) -> tuple[np.ndarray, ...]:
    """Copy primary state excluding scheduler-refreshed derived fields."""
    state = _snapshot(session)
    return state[:4] + state[5:7]


def _rng_snapshot(session: Any, rng_states: Any) -> np.ndarray:
    """Synchronize explicitly before reading a resident RNG sidecar."""
    wp = pytest.importorskip("warp")
    wp.synchronize_device(session.particles.masses.device)
    return rng_states.numpy().copy()


def _inventory(session: Any) -> np.ndarray:
    """Return independent concentration-weighted particle-plus-gas inventory."""
    state = _snapshot(session)
    masses, concentration, _, gas, *_ = state
    # ``_snapshot`` is the explicit readback boundary for all resident data.
    volume = session.particles.volume.numpy()
    return volume[:, None] * (
        np.sum(masses * concentration[:, :, None], axis=1) + gas
    )


def _communication_configuration(session: Any, wp: Any) -> Any:
    """Build a closed zero-edge GAS map on the resident device."""
    device = session.particles.masses.device
    return CommunicationConfiguration(
        CommunicationMap(
            CommunicationMapForm.ARBITRARY_PAIRS,
            CommunicationTransportMode.GAS,
            0,
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.int32, device=device),
            wp.zeros(0, dtype=wp.float64, device=device),
        ),
        PrescribedVolumeUpdate(None),
        (
            CommunicationResourceShape(
                "edge_rates", wp.float64, CommunicationShapeKind.E
            ),
        ),
    )


@cache
def _resident_graph() -> tuple[Any, Any, dict[str, Any]]:
    """Reuse one resolver-produced complete graph across repeated seed rows."""
    from particula.execution import process_graph

    def node(node_id: str) -> ProcessNode:
        """Build one canonical resident node with its capability requirements."""
        schema = next(
            item
            for item in process_graph._NODE_CATALOGUE
            if item.node_id == node_id
        )
        requirements = (
            next(
                declaration.requirements
                for declaration in CONDENSATION_CAPABILITY_MATRIX.declarations
                if declaration.process == CONDENSATION_PROCESS
            )
            if node_id == "condensation"
            else CapabilityRequirements(frozenset())
        )
        return ProcessNode(
            schema.node_id,
            schema.kind,
            schema.process,
            requirements,
            schema.resources,
            schema.invalidates,
        )

    nodes = tuple(node(node_id) for node_id in _NODE_IDS)
    dependencies = (
        (DependencyEdge("communication", "volume_evolution"),)
        + tuple(
            DependencyEdge("volume_evolution", node_id)
            for node_id in _NODE_IDS
            if node_id not in {"communication", "volume_evolution"}
        )
        + tuple(
            DependencyEdge(node_id, "diagnostics")
            for node_id in (
                "condensation",
                "brownian_coagulation",
                "dilution",
                "wall_loss",
                "nucleation",
            )
        )
    )
    schedule = resolve_timestep_schedule(
        TimestepPlan(nodes, dependencies),
        EnabledNodeSelection(frozenset(_NODE_IDS)),
        SchedulerProfile(
            NucleationCondensationDirection.CONDENSATION_THEN_NUCLEATION
        ),
    )
    graph = schedule.source_graph
    assert graph is not None
    return graph, schedule, {node.node_id: node for node in graph.nodes}


def _diagnostics_plan(
    session: Any,
    registry: GPUResourceRegistry,
    graph: Any,
    schedule: Any,
    by_id: dict[str, Any],
    wp: Any,
) -> ResidentDiagnosticsPlan:
    """Build the exact closed diagnostics protocol for dynamic dimensions."""
    boxes = session.dimensions.n_boxes
    species = session.dimensions.n_species
    device = session.particles.masses.device

    def matrix() -> Any:
        """Allocate one diagnostics matrix on the resident device."""
        return wp.zeros((boxes, species), dtype=wp.float64, device=device)

    total_mass = matrix()
    registrations = (
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT, matrix()
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT, matrix()
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.TOTAL_SPECIES_MASS, total_mass
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION,
            wp.zeros(boxes, dtype=wp.float64, device=device),
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
            matrix(),
            energy_transfer=matrix(),
        ),
        ResidentDiagnosticRegistration(
            ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
            matrix(),
            baseline_total_mass=matrix(),
            source_ledger=matrix(),
            sink_ledger=matrix(),
        ),
    )
    return ResidentDiagnosticsPlan(
        session,
        registry,
        graph,
        schedule,
        by_id["diagnostics"],
        registrations,
    )


def _scheduler_request(
    session: Any,
    registry: GPUResourceRegistry,
    guard: ResidentStepGuard,
    duration: float,
    selected: tuple[int, ...] | None,
) -> tuple[ResidentSimulationRequest, Any, Any]:
    """Build one complete scheduler request with isolated wall-loss physics."""
    wp = pytest.importorskip("warp")
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    graph, schedule, by_id = _resident_graph()
    device = session.particles.masses.device
    dimensions = session.dimensions
    stream = session.metadata.stream
    manifest = tuple(zip(stream.logical_box_ids, stream.lanes, strict=True))
    _, cpu_gas, cpu_environment = _cpu_carriers(manifest)

    communication = registry.acquire_communication(
        _communication_configuration(session, wp)
    )
    condensation_resources = registry.acquire_condensation()
    coagulation_resources = registry.acquire_coagulation(1)
    wall_loss_resources = registry.acquire_wall_loss()
    nucleation_resources = registry.acquire_nucleation()

    thermodynamics = ThermodynamicsConfig(
        modes=wp.zeros(dimensions.n_species, dtype=wp.int32, device=device),
        parameters=wp.array(
            np.tile(
                np.array([800.0, 0.0, 0.0, 0.0], dtype=np.float64),
                (dimensions.n_species, 1),
            ),
            dtype=wp.float64,
            device=device,
        ),
        molar_mass_reference=wp.array(
            cpu_gas.molar_mass, dtype=wp.float64, device=device
        ),
    )
    condensation = WarpCondensationExecutionState(
        WarpCondensationState(
            CondensationExecutionConfig(
                CondensationConfiguration(
                    CondensationExecutionMode.EQUAL_STEP,
                    False,
                    CondensationActivityMode.IDEAL,
                    CondensationSurfaceMode.STATIC,
                )
            ),
            session.particles,
            session.gas,
            session.environment,
            thermodynamics,
            scratch_buffers=condensation_resources.scratch_buffers,
        ),
        duration,
    )
    coagulation = ResidentBrownianCoagulationExecutionState(
        WarpBrownianCoagulationExecutionState(
            WarpBrownianCoagulationState(
                BrownianCoagulationConfig(),
                session.particles,
                None,
                None,
                duration,
                collision_pairs=coagulation_resources.collision_pairs,
                n_collisions=coagulation_resources.n_collisions,
                rng_states=coagulation_resources.rng_states,
                initialize_rng=False,
                environment=session.environment,
            )
        ),
        session,
        registry,
        coagulation_resources,
    )
    environment_update = ResidentEnvironmentUpdateRequest(
        session,
        registry,
        graph,
        by_id["environment_update"],
        wp.array(cpu_environment.temperature, dtype=wp.float64, device=device),
        wp.array(cpu_environment.pressure, dtype=wp.float64, device=device),
    )
    gas_update = ResidentGasUpdateRequest(
        session,
        registry,
        graph,
        by_id["gas_update"],
        wp.array(cpu_gas.concentration, dtype=wp.float64, device=device),
    )
    request = ResidentSimulationRequest(
        session,
        registry,
        guard,
        graph,
        schedule,
        thermodynamics,
        condensation,
        coagulation,
        ResidentDilutionRequest(session, registry, 0.0, duration),
        ResidentWallLossRequest(
            session,
            registry,
            wall_loss_resources,
            NeutralWallLossConfig("spherical", 1.0, chamber_radius=1.0),
            duration,
            enabled_box_indices=selected,
        ),
        ResidentNucleationRequest(
            session,
            registry,
            nucleation_resources,
            object(),
            duration,
            object(),
        ),
        _diagnostics_plan(session, registry, graph, schedule, by_id, wp),
        environment_update,
        gas_update,
        ResidentCommunicationRequest(
            session,
            registry,
            graph,
            communication,
            by_id["communication"],
            by_id["volume_evolution"],
            duration,
        ),
    )
    return request, wall_loss_resources, coagulation_resources


def _wall_loss(
    session: Any,
    registry: GPUResourceRegistry,
    guard: ResidentStepGuard,
    duration: float,
    selected: tuple[int, ...] | None = None,
) -> Any:
    """Execute real neutral wall loss through the complete resident scheduler."""
    request, wall_loss, coagulation = _scheduler_request(
        session, registry, guard, duration, selected
    )
    assert wall_loss.rng_states is not coagulation.rng_states
    ResidentSimulationScheduler(request).execute(duration)
    request.guard.assert_step_closed()
    return wall_loss


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_multi_box_zero_duration_matches_independent_sessions_and_conserves_inventory(
    resident_factory: Any,
) -> None:
    """Zero-duration selected dispatch preserves four logical rows and inventory."""
    manifest = tuple((f"box-{index}", index) for index in range(4))
    multi, multi_registry, multi_guard = resident_factory("cpu", manifest)
    initial_inventory = _inventory(multi)
    before = _primary_snapshot(multi)
    wall_before = _rng_snapshot(
        multi, multi_registry.acquire_wall_loss().rng_states
    )
    coagulation_before = _rng_snapshot(
        multi, multi_registry.acquire_coagulation(1).rng_states
    )
    resources = _wall_loss(
        multi, multi_registry, multi_guard, 0.0, (0, 1, 2, 3)
    )
    assert resources.rng_states is multi_registry.acquire_wall_loss().rng_states
    after = _primary_snapshot(multi)
    for actual, expected in zip(after, before, strict=True):
        npt.assert_allclose(
            actual, expected, rtol=PARITY_RTOL, atol=PARITY_ATOL
        )
    npt.assert_allclose(
        _inventory(multi),
        initial_inventory,
        rtol=INVENTORY_RTOL,
        atol=INVENTORY_ATOL,
    )
    npt.assert_equal(_rng_snapshot(multi, resources.rng_states), wall_before)
    npt.assert_equal(
        _rng_snapshot(multi, multi_registry.acquire_coagulation(1).rng_states),
        coagulation_before,
    )
    for logical_id, _ in manifest:
        single, single_registry, single_guard = resident_factory(
            "cpu", ((logical_id, 0),)
        )
        _wall_loss(single, single_registry, single_guard, 0.0, (0,))
        lane = _lane(multi, logical_id)
        for actual, expected in zip(
            _snapshot(multi), _snapshot(single), strict=True
        ):
            npt.assert_allclose(
                actual[lane], expected[0], rtol=PARITY_RTOL, atol=PARITY_ATOL
            )


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_multi_box_logical_id_permutation_addition_and_wall_loss_selection_are_isolated(
    resident_factory: Any,
) -> None:
    """Positive scheduler work is stable across logical reorder and addition."""
    reference, reference_registry, reference_guard = resident_factory(
        "cpu", (("box-a", 0), ("box-b", 1), ("box-c", 2), ("box-d", 3))
    )
    candidate, candidate_registry, candidate_guard = resident_factory(
        "cpu",
        (("box-c", 2), ("extra", 4), ("box-a", 0), ("box-d", 3), ("box-b", 1)),
    )
    _wall_loss(
        reference, reference_registry, reference_guard, 1.0, (0, 1, 2, 3)
    )
    reference_wall = _rng_snapshot(
        reference, reference_registry.acquire_wall_loss().rng_states
    )
    before = _primary_snapshot(candidate)
    wall_before = _rng_snapshot(
        candidate, candidate_registry.acquire_wall_loss().rng_states
    )
    coagulation_before = _rng_snapshot(
        candidate, candidate_registry.acquire_coagulation(1).rng_states
    )
    _wall_loss(
        candidate, candidate_registry, candidate_guard, 1.0, (0, 2, 3, 4)
    )
    after = _primary_snapshot(candidate)
    extra_lane = _lane(candidate, "extra")
    for actual, expected in zip(after, before, strict=True):
        npt.assert_equal(actual[extra_lane], expected[extra_lane])
    wall_after = _rng_snapshot(
        candidate, candidate_registry.acquire_wall_loss().rng_states
    )
    assert wall_after[extra_lane] == wall_before[extra_lane]
    npt.assert_equal(
        _rng_snapshot(
            candidate, candidate_registry.acquire_coagulation(1).rng_states
        ),
        coagulation_before,
    )
    for logical_id in ("box-a", "box-b", "box-c", "box-d"):
        assert (
            wall_after[_lane(candidate, logical_id)]
            == reference_wall[_lane(reference, logical_id)]
        )
        for actual, expected in zip(
            _snapshot(candidate), _snapshot(reference), strict=True
        ):
            npt.assert_allclose(
                actual[_lane(candidate, logical_id)],
                expected[_lane(reference, logical_id)],
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )
    empty, empty_registry, empty_guard = resident_factory(
        "cpu", (("box-a", 0), ("box-b", 1), ("box-c", 2), ("box-d", 3))
    )
    before_empty = _primary_snapshot(empty)
    rng_before_empty = _rng_snapshot(
        empty, empty_registry.acquire_wall_loss().rng_states
    )
    _wall_loss(empty, empty_registry, empty_guard, 1.0, ())
    for actual, expected in zip(
        _primary_snapshot(empty), before_empty, strict=True
    ):
        npt.assert_equal(actual, expected)
    npt.assert_equal(
        _rng_snapshot(empty, empty_registry.acquire_wall_loss().rng_states),
        rng_before_empty,
    )


@pytest.mark.warp
@pytest.mark.stochastic
def test_resident_scheduler_streams_continue_per_logical_box_without_no_work_consumption(
    resident_factory: Any,
) -> None:
    """Selected active rows advance their wall-loss stream; free rows do not."""
    session, registry, guard = resident_factory(
        "cpu", (("active", 0), ("free", 1), ("other", 2), ("one", 3))
    )
    resources = registry.acquire_wall_loss()
    coagulation = registry.acquire_coagulation(1)
    before = _rng_snapshot(session, resources.rng_states)
    coagulation_before = _rng_snapshot(session, coagulation.rng_states)
    _wall_loss(session, registry, guard, 1.0, (0, 1))
    after = _rng_snapshot(session, resources.rng_states)
    assert after[0] != before[0]
    assert after[1] == before[1]
    assert resources.rng_states is registry.acquire_wall_loss().rng_states
    assert resources.rng_states is not coagulation.rng_states
    npt.assert_equal(
        _rng_snapshot(session, coagulation.rng_states), coagulation_before
    )


@pytest.mark.warp
@pytest.mark.stochastic
def test_resident_wall_loss_removal_matches_cpu_binomial_aggregate() -> None:
    """Fresh Warp CPU streams produce finite bounded aggregate removal evidence."""
    from particula.dynamics.properties.wall_loss_coefficient import (
        get_spherical_wall_loss_coefficient_via_system_state,
    )

    time_step = 1.0
    density = np.array([1000.0, 1200.0], dtype=np.float64)
    masses = np.array([3.29e-18, 6.58e-18], dtype=np.float64)
    particle_volume = np.sum(masses / density)
    particle_radius = (3.0 * particle_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    coefficient = get_spherical_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=1.0,
        particle_radius=particle_radius,
        particle_density=float(np.sum(masses) / particle_volume),
        temperature=298.29,
        pressure=100996.0,
        chamber_radius=1.0,
    )
    removal_probability = 1.0 - np.exp(-float(coefficient) * time_step)
    removed = 0
    trials = 0
    for seed in range(100):
        with _temporary_binding("cpu", (("box", 0),), seed) as binding:
            session, registry, guard = binding
            before = _snapshot(session)[1]
            _wall_loss(session, registry, guard, time_step, (0,))
            after = _snapshot(session)[1]
            removed += int(np.count_nonzero((before > 0.0) & (after == 0.0)))
            trials += int(np.count_nonzero(before > 0.0))
    expected = trials * removal_probability
    bound = max(
        3.0
        * np.sqrt(trials * removal_probability * (1.0 - removal_probability)),
        1.0,
    )
    assert abs(removed - expected) <= bound


@pytest.mark.warp
@pytest.mark.cuda
def test_resident_wall_loss_cuda_smoke_has_finite_bounded_removal() -> None:
    """Optional CUDA lifecycle smoke keeps removal counts finite and bounded."""
    removed = 0
    trials = 0
    for seed in range(12):
        with _temporary_binding("cuda", (("box", 0),), seed) as binding:
            session, registry, guard = binding
            before = _snapshot(session)[1]
            _wall_loss(session, registry, guard, 1.0, (0,))
            after = _snapshot(session)[1]
            assert np.all(np.isfinite(after))
            removed += int(np.count_nonzero((before > 0.0) & (after == 0.0)))
            trials += int(np.count_nonzero(before > 0.0))
    assert 0 <= removed <= trials
