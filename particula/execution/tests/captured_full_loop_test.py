"""Three-way evidence boundaries for resident graph-capture full loops.

Warp CPU supplies the uncaptured resident baseline. Native CUDA capture is
optional and is skipped before qualification when the device or Warp capture
API is unavailable; it is never emulated on the CPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution import Backend, Device
from particula.execution.communication import CommunicationTransportMode
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
from particula.execution.tests.full_loop_test import _build_loop_fixture
from particula.gpu.tests.cuda_availability import (
    CUDA_SKIP_REASON,
    cuda_available,
)

PARITY_RTOL = 1e-12
PARITY_ATOL = 1e-30


@pytest.fixture(autouse=True)
def _remove_resolver_schedule_registrations() -> Any:
    """Keep locally constructed schedules isolated between test cases."""
    import particula.execution.scheduler as scheduler_module

    schedules = scheduler_module._RESOLVER_SCHEDULES
    initial_schedules = list(schedules)
    schedules.clear()
    yield
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
    if not all(
        callable(getattr(wp, name, None))
        for name in ("capture_begin", "capture_end")
    ):
        pytest.skip("Warp capture API unavailable")
    return wp


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
        assert captured.handle is native.handle
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
