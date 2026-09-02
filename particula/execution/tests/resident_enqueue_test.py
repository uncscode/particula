"""Contract tests for READY-only resident enqueue preparation."""

from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from particula.execution import Backend, Device
from particula.execution.gpu_session import ResidentStepGuard
from particula.execution.graph_capture import (
    GraphCaptureAvailability,
    GraphCaptureCapability,
    GraphCaptureCompatibility,
    GraphCaptureDriftReason,
    GraphCaptureLifecycleState,
    ResidentGraphCaptureBinding,
    _attach_resident_graph_capture_binding,
    create_graph_capture_lifecycle,
    create_resident_graph_capture_signature,
)
from particula.execution.resident_enqueue import (
    PreparedResidentTimestep,
    prepare_resident_timestep,
)


def _ready_request(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build an attached READY request without completing capture."""
    pytest.importorskip("warp")
    from particula.execution.communication import CommunicationTransportMode
    from particula.execution.tests.full_loop_test import _build_loop_fixture

    fixture = _build_loop_fixture(monkeypatch, CommunicationTransportMode.GAS)
    request = fixture.request
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            request.session.metadata.device,
            GraphCaptureAvailability.AVAILABLE,
        ),
        create_resident_graph_capture_signature(request),
    )
    binding = ResidentGraphCaptureBinding(
        request,
        request.session,
        request.registry,
        request.guard,
        lifecycle,
    )
    _attach_resident_graph_capture_binding(request, binding)
    return fixture


@pytest.mark.warp
def test_prepare_retains_ready_metadata_without_opening_a_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid READY declaration returns frozen identity-only metadata."""
    fixture = _ready_request(monkeypatch)
    request = fixture.request
    binding = request.graph_capture_binding
    assert binding is not None
    begin_calls: list[object] = []

    def reject_step(*args: object, **kwargs: object) -> None:
        """Record forbidden token entry during metadata preparation."""
        begin_calls.append((args, kwargs))
        raise AssertionError("preparation must not open a resident step")

    monkeypatch.setattr(request.guard, "begin_step", reject_step)

    prepared = prepare_resident_timestep(request, 0.0)

    assert prepared.request is request
    assert prepared.binding is binding
    assert prepared.lifecycle is binding.lifecycle
    assert prepared.signature is binding.lifecycle.signature
    assert prepared.session is request.session
    assert prepared.registry is request.registry
    assert prepared.guard is request.guard
    assert prepared.device is binding.lifecycle.signature.device
    assert prepared.dimensions is binding.lifecycle.signature.dimensions
    assert prepared.graph is request.graph
    assert prepared.schedule is request.schedule
    assert prepared.ordered_node_ids is request.schedule.ordered_node_ids
    assert prepared.primary_arrays is binding.lifecycle.signature.primary_arrays
    assert prepared.resource_views is binding.lifecycle.signature.resource_views
    assert all(
        actual is expected
        for actual, expected in zip(
            prepared.primary_arrays,
            binding.lifecycle.signature.primary_arrays,
            strict=True,
        )
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            prepared.resource_views,
            binding.lifecycle.signature.resource_views,
            strict=True,
        )
    )
    assert prepared.duration == 0.0
    assert begin_calls == []
    assert request.guard.completed_steps == 0
    assert binding.lifecycle.state is GraphCaptureLifecycleState.READY
    with pytest.raises(FrozenInstanceError):
        prepared.duration = 1.0  # type: ignore[assignment, misc]
    assert prepare_resident_timestep(request, 0.0) != prepared


@pytest.mark.warp
@pytest.mark.parametrize("duration", (None, True, "0", float("nan"), -1.0))
def test_prepare_rejects_invalid_duration_without_mutating_ready_lifecycle(
    monkeypatch: pytest.MonkeyPatch, duration: object
) -> None:
    """Duration preflight leaves the attached declaration READY and unchanged."""
    fixture = _ready_request(monkeypatch)
    request = fixture.request
    binding = request.graph_capture_binding
    assert binding is not None

    with pytest.raises((TypeError, ValueError), match="duration must"):
        prepare_resident_timestep(request, duration)

    assert binding.lifecycle.state is GraphCaptureLifecycleState.READY
    assert request.graph_capture_binding is binding
    assert request.guard.completed_steps == 0


@pytest.mark.warp
def test_prepare_rechecks_signature_after_metadata_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift detected after shared validation returns no prepared carrier."""
    fixture = _ready_request(monkeypatch)
    request = fixture.request
    binding = request.graph_capture_binding
    assert binding is not None
    import particula.execution.graph_capture as graph_capture

    calls: list[object] = []

    def compare_after_validation(
        signature: object, candidate: object
    ) -> object:
        """Accept the first comparison and report drift at the return guard."""
        calls.append((signature, candidate))
        return GraphCaptureCompatibility(
            compatible=len(calls) == 1,
            reason=(
                None
                if len(calls) == 1
                else GraphCaptureDriftReason.SCHEDULE_ORDER
            ),
        )

    monkeypatch.setattr(
        graph_capture,
        "compare_resident_graph_capture_signature",
        compare_after_validation,
    )

    with pytest.raises(ValueError, match="signature is incompatible"):
        prepare_resident_timestep(request, 0.0)

    assert len(calls) == 2
    assert binding.lifecycle.state is GraphCaptureLifecycleState.READY
    assert request.graph_capture_binding is binding
    assert request.guard.completed_steps == 0


@pytest.mark.warp
def test_prepare_rejects_open_step_owned_by_another_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry-wide open step blocks preparation through a closed guard."""
    fixture = _ready_request(monkeypatch)
    request = fixture.request
    binding = request.graph_capture_binding
    assert binding is not None
    other_guard = ResidentStepGuard(request.session, request.registry)
    token = other_guard.begin_step(0.0)

    with pytest.raises(RuntimeError, match="resident timestep is open"):
        prepare_resident_timestep(request, 0.0)

    assert request.guard._open_token is None
    assert request.guard.completed_steps == 0
    assert other_guard._open_token is token
    assert request.registry._open_step_token is token
    assert fixture.trace == []
    assert binding.lifecycle.state is GraphCaptureLifecycleState.READY
    other_guard.complete_step(token)


@pytest.mark.warp
def test_direct_prepared_carrier_rejects_detached_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct carrier construction cannot bypass exact request attachment."""
    fixture = _ready_request(monkeypatch)
    request = fixture.request
    binding = request.graph_capture_binding
    assert binding is not None
    signature = binding.lifecycle.signature
    object.__setattr__(request, "graph_capture_binding", None)

    with pytest.raises(ValueError, match="binding identities do not match"):
        PreparedResidentTimestep(
            request=request,
            binding=binding,
            lifecycle=binding.lifecycle,
            signature=signature,
            session=request.session,
            registry=request.registry,
            guard=request.guard,
            device=signature.device,
            dimensions=signature.dimensions,
            graph=request.graph,
            schedule=request.schedule,
            ordered_node_ids=request.schedule.ordered_node_ids,
            duration=0.0,
            primary_arrays=signature.primary_arrays,
            resource_views=signature.resource_views,
        )


@pytest.mark.warp
def test_prepare_rejects_ready_capability_device_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """READY preparation rejects a capability for another resident device."""
    fixture = _ready_request(monkeypatch)
    request = fixture.request
    binding = request.graph_capture_binding
    assert binding is not None
    capability = GraphCaptureCapability(
        Device(Backend.WARP, "cuda:99"),
        GraphCaptureAvailability.AVAILABLE,
    )
    binding._lifecycle = replace(binding.lifecycle, capability=capability)

    with pytest.raises(ValueError, match="capability device does not match"):
        prepare_resident_timestep(request, 0.0)

    assert binding.lifecycle.state is GraphCaptureLifecycleState.READY
    assert request.guard.completed_steps == 0
