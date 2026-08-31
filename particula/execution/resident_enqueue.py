"""Prepare READY resident graph metadata for later enqueue phases.

This concrete direct-import-only P1 boundary validates and freezes READY-state
identity metadata only. It does not construct executors, capture, enqueue,
dispatch, acquire resources, inspect payloads, transfer, synchronize, mutate a
lifecycle, or fall back.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

from particula.execution import _isfinite_real

if TYPE_CHECKING:
    from particula.execution.graph_capture import (
        GraphCaptureLifecycle,
        ResidentGraphCaptureBinding,
        ResidentGraphCaptureSignature,
    )
    from particula.execution.resident_scheduler import ResidentSimulationRequest
    from particula.execution.scheduler import ResolvedTimestepSchedule


def _request_type() -> type[object]:
    """Return the concrete resident simulation request type lazily.

    Returns:
        The direct-import-only resident simulation request type.
    """
    from particula.execution.resident_scheduler import ResidentSimulationRequest

    return ResidentSimulationRequest


def _graph_capture_types() -> tuple[type[object], type[object], type[object]]:
    """Return concrete graph-capture carrier types lazily.

    Returns:
        Exact binding, lifecycle, and signature types in that order.
    """
    from particula.execution.graph_capture import (
        GraphCaptureLifecycle,
        ResidentGraphCaptureBinding,
        ResidentGraphCaptureSignature,
    )

    return (
        ResidentGraphCaptureBinding,
        GraphCaptureLifecycle,
        ResidentGraphCaptureSignature,
    )


def _validate_ready_attachment(
    request: object,
    binding: object,
    lifecycle: object,
    signature: object,
    session: object,
    registry: object,
    guard: object,
) -> None:
    """Require one exact, closed, active READY resident attachment.

    This check is shared by direct prepared-carrier construction and preparation
    return guards so neither path can retain stale attachment metadata.
    """
    from particula.execution.graph_capture import GraphCaptureLifecycleState

    if (
        request.graph_capture_binding is not binding
        or binding._request is not request
        or binding._session is not session
        or binding._registry is not registry
        or binding._guard is not guard
        or binding.lifecycle is not lifecycle
        or lifecycle.signature is not signature
        or request.session is not session
        or request.registry is not registry
        or request.guard is not guard
        or guard._session is not session
        or guard._registry is not registry
        or registry._session is not session
    ):
        raise ValueError("graph-capture binding identities do not match.")
    guard.assert_step_closed()
    registry.validate_pinned_session(session)
    if session.lifecycle.name != "ACTIVE":
        raise ValueError("resident session must be ACTIVE.")
    if lifecycle.state is not GraphCaptureLifecycleState.READY:
        raise ValueError("graph capture must be ready for preparation.")
    if lifecycle.capability.device != session.metadata.device:
        raise ValueError("capability device does not match session.")


@dataclass(frozen=True, eq=False)
class PreparedResidentTimestep:
    """Retain exact READY-bound metadata for a future resident enqueue.

    This frozen, identity-semantic carrier is preparation output only. It
    retains existing host metadata and published resources without copying or
    acquiring them, and it does not authorize capture, enqueue, or dispatch.

    Attributes:
        request: Exact complete resident request retained by the binding.
        binding: Exact READY graph-capture binding for ``request``.
        lifecycle: Exact READY lifecycle retained by ``binding``.
        signature: Exact lifecycle signature used for compatibility checks.
        session: Exact resident session shared by request and binding.
        registry: Exact resource registry shared by request and binding.
        guard: Exact closed resident-step guard shared by request and binding.
        device: Exact signature device.
        dimensions: Exact signature resident dimensions.
        graph: Exact resolved graph retained by ``request``.
        schedule: Exact resolved schedule retained by ``request``.
        ordered_node_ids: Exact canonical schedule node-ID tuple.
        duration: Original finite, nonnegative timestep duration.
        primary_arrays: Exact signature primary-array tuple.
        resource_views: Exact signature published-resource tuple.
    """

    request: "ResidentSimulationRequest"
    binding: "ResidentGraphCaptureBinding"
    lifecycle: "GraphCaptureLifecycle"
    signature: "ResidentGraphCaptureSignature"
    session: object
    registry: object
    guard: object
    device: object
    dimensions: object
    graph: object
    schedule: "ResolvedTimestepSchedule"
    ordered_node_ids: tuple[object, ...]
    duration: Real
    primary_arrays: tuple[object, ...]
    resource_views: tuple[object, ...]

    def __post_init__(self) -> None:  # noqa: C901
        """Validate exact carriers, READY state, and retained identities.

        Raises:
            TypeError: If a carrier, tuple, or duration has an invalid type.
            ValueError: If duration is invalid, identities drift, or lifecycle
                state is not READY.
        """
        from particula.execution import Device
        from particula.execution.gpu_resources import GPUResourceRegistry
        from particula.execution.gpu_session import (
            ResidentDimensions,
            ResidentSession,
            ResidentStepGuard,
        )
        from particula.execution.process_graph import ResolvedProcessGraph
        from particula.execution.scheduler import ResolvedTimestepSchedule

        request_type = _request_type()
        binding_type, lifecycle_type, signature_type = _graph_capture_types()
        if type(self.request) is not request_type:
            raise TypeError(
                "request must be an exact ResidentSimulationRequest."
            )
        if type(self.binding) is not binding_type:
            raise TypeError(
                "binding must be an exact ResidentGraphCaptureBinding."
            )
        if type(self.lifecycle) is not lifecycle_type:
            raise TypeError("lifecycle must be an exact GraphCaptureLifecycle.")
        if type(self.signature) is not signature_type:
            raise TypeError(
                "signature must be an exact ResidentGraphCaptureSignature."
            )
        exact = (
            (self.session, ResidentSession, "session"),
            (self.registry, GPUResourceRegistry, "registry"),
            (self.guard, ResidentStepGuard, "guard"),
            (self.device, Device, "device"),
            (self.dimensions, ResidentDimensions, "dimensions"),
            (self.graph, ResolvedProcessGraph, "graph"),
            (self.schedule, ResolvedTimestepSchedule, "schedule"),
        )
        for value, expected, name in exact:
            if type(value) is not expected:
                raise TypeError(f"{name} must be an exact {expected.__name__}.")
        if type(self.ordered_node_ids) is not tuple:
            raise TypeError("ordered_node_ids must be an exact tuple.")
        if type(self.primary_arrays) is not tuple:
            raise TypeError("primary_arrays must be an exact tuple.")
        if type(self.resource_views) is not tuple:
            raise TypeError("resource_views must be an exact tuple.")
        if isinstance(self.duration, bool) or not isinstance(
            self.duration, Real
        ):
            raise TypeError("duration must be a non-boolean real.")
        if not _isfinite_real(self.duration) or self.duration < 0:
            raise ValueError("duration must be finite and nonnegative.")
        if (
            self.binding._request is not self.request
            or self.binding._session is not self.session
            or self.binding._registry is not self.registry
            or self.binding._guard is not self.guard
            or self.binding.lifecycle is not self.lifecycle
            or self.lifecycle.signature is not self.signature
            or self.request.session is not self.session
            or self.request.registry is not self.registry
            or self.request.guard is not self.guard
            or self.signature.request is not self.request
            or self.signature.session is not self.session
            or self.signature.device is not self.device
            or self.signature.dimensions is not self.dimensions
            or self.request.graph is not self.graph
            or self.request.schedule is not self.schedule
            or self.schedule.ordered_node_ids is not self.ordered_node_ids
            or self.signature.primary_arrays is not self.primary_arrays
            or self.signature.resource_views is not self.resource_views
            or self.signature.graph[0] is not self.graph
            or self.signature.schedule[0] is not self.schedule
            or self.signature.schedule_order[0] is not self.ordered_node_ids
        ):
            raise ValueError(
                "prepared resident timestep identities do not match."
            )
        _validate_ready_attachment(
            self.request,
            self.binding,
            self.lifecycle,
            self.signature,
            self.session,
            self.registry,
            self.guard,
        )


def prepare_resident_timestep(  # noqa: C901
    request: object, duration: object
) -> PreparedResidentTimestep:
    """Validate and freeze one READY resident timestep without side effects.

    The direct-only preparation boundary performs shared read-only metadata
    validation and retains identities in a frozen carrier. It does not construct
    executors, open a guard token, acquire resources, inspect payloads, capture,
    enqueue, dispatch, transfer, synchronize, mutate lifecycle state, or fall
    back.

    Args:
        request: Exact attached resident simulation request.
        duration: Non-boolean finite, nonnegative timestep duration.

    Returns:
        A frozen identity-only prepared timestep.

    Raises:
        TypeError: If a request, attachment, retained carrier, or duration has
            an inexact or invalid type.
        ValueError: If duration, ownership, lifecycle state, signature, or
            complete-loop metadata is invalid.
    """
    if type(request) is not _request_type():
        raise TypeError("request must be an exact ResidentSimulationRequest.")
    request_any: Any = request
    if isinstance(duration, bool) or not isinstance(duration, Real):
        raise TypeError("duration must be a non-boolean real.")
    if not _isfinite_real(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative.")
    from particula.execution.gpu_resources import GPUResourceRegistry
    from particula.execution.gpu_session import (
        ResidentSession,
        ResidentStepGuard,
    )

    if type(request_any.session) is not ResidentSession:
        raise TypeError("session must be an exact ResidentSession.")
    if type(request_any.registry) is not GPUResourceRegistry:
        raise TypeError("registry must be an exact GPUResourceRegistry.")
    if type(request_any.guard) is not ResidentStepGuard:
        raise TypeError("guard must be an exact ResidentStepGuard.")
    binding = request_any.graph_capture_binding
    binding_type, lifecycle_type, signature_type = _graph_capture_types()
    if type(binding) is not binding_type:
        raise TypeError("graph_capture_binding must be an exact binding.")
    binding_any: Any = binding
    lifecycle_value = binding_any.lifecycle
    if type(lifecycle_value) is not lifecycle_type:
        raise TypeError("lifecycle must be an exact GraphCaptureLifecycle.")
    lifecycle: Any = lifecycle_value
    signature_value = lifecycle.signature
    if type(signature_value) is not signature_type:
        raise TypeError(
            "signature must be an exact ResidentGraphCaptureSignature."
        )
    signature: Any = signature_value
    from particula.execution.graph_capture import (
        compare_resident_graph_capture_signature,
    )
    from particula.execution.resident_scheduler import (
        _validate_complete_resident_timestep_metadata,
    )

    _validate_ready_attachment(
        request_any,
        binding_any,
        lifecycle,
        signature,
        request_any.session,
        request_any.registry,
        request_any.guard,
    )
    compatibility = compare_resident_graph_capture_signature(
        cast(Any, signature), cast(Any, request_any)
    )
    if not compatibility.compatible:
        raise ValueError("resident graph-capture signature is incompatible.")
    _validate_complete_resident_timestep_metadata(request_any, duration)
    compatibility = compare_resident_graph_capture_signature(
        cast(Any, signature), cast(Any, request_any)
    )
    if not compatibility.compatible:
        raise ValueError("resident graph-capture signature is incompatible.")
    _validate_ready_attachment(
        request_any,
        binding_any,
        lifecycle,
        signature,
        request_any.session,
        request_any.registry,
        request_any.guard,
    )
    return PreparedResidentTimestep(
        request=request_any,
        binding=binding_any,
        lifecycle=lifecycle,
        signature=signature,
        session=request_any.session,
        registry=request_any.registry,
        guard=request_any.guard,
        device=signature.device,
        dimensions=signature.dimensions,
        graph=request_any.graph,
        schedule=request_any.schedule,
        ordered_node_ids=request_any.schedule.ordered_node_ids,
        duration=duration,
        primary_arrays=signature.primary_arrays,
        resource_views=signature.resource_views,
    )
