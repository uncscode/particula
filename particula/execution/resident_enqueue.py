"""Prepare READY resident graph metadata for later enqueue phases.

This concrete direct-import-only P1 boundary validates and freezes identity
metadata only. It does not capture, enqueue, dispatch, acquire resources,
inspect payloads, transfer, synchronize, mutate a lifecycle, or fall back.
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
    """Lazily return the concrete resident simulation request type."""
    from particula.execution.resident_scheduler import ResidentSimulationRequest

    return ResidentSimulationRequest


def _graph_capture_types() -> tuple[type[object], type[object], type[object]]:
    """Lazily return the concrete graph-capture carrier types."""
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


@dataclass(frozen=True, eq=False)
class PreparedResidentTimestep:
    """Retain exact READY-bound metadata for a future resident enqueue.

    All fields retain existing host metadata and published resources by
    identity.
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
        """Require exact carriers and their retained identity links."""
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
        from particula.execution.graph_capture import GraphCaptureLifecycleState

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
        if self.lifecycle.state is not GraphCaptureLifecycleState.READY:
            raise ValueError(
                "prepared resident timestep requires ready lifecycle."
            )


def prepare_resident_timestep(
    request: object, duration: object
) -> PreparedResidentTimestep:
    """Validate and freeze one READY resident timestep without side effects.

    Args:
        request: Exact attached resident simulation request.
        duration: Non-boolean finite, nonnegative timestep duration.

    Returns:
        A frozen identity-only prepared timestep.
    """
    if type(request) is not _request_type():
        raise TypeError("request must be an exact ResidentSimulationRequest.")
    request = cast("ResidentSimulationRequest", request)
    if isinstance(duration, bool) or not isinstance(duration, Real):
        raise TypeError("duration must be a non-boolean real.")
    if not _isfinite_real(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative.")
    binding = request.graph_capture_binding
    binding_type, _, _ = _graph_capture_types()
    if type(binding) is not binding_type:
        raise TypeError("graph_capture_binding must be an exact binding.")
    binding = cast(Any, binding)
    registry = cast(Any, request.registry)
    if (
        binding._request is not request
        or binding._session is not request.session
        or binding._registry is not request.registry
        or binding._guard is not request.guard
        or request.guard._session is not request.session
        or request.guard._registry is not request.registry
        or registry._session is not request.session
    ):
        raise ValueError(
            "resident graph-capture binding identities do not match."
        )
    request.guard.assert_step_closed()
    registry.validate_pinned_session(request.session)
    from particula.execution.graph_capture import (
        GraphCaptureLifecycleState,
        compare_resident_graph_capture_signature,
    )
    from particula.execution.resident_scheduler import (
        _validate_complete_resident_timestep_metadata,
    )

    lifecycle = binding.lifecycle
    if lifecycle.state is not GraphCaptureLifecycleState.READY:
        raise ValueError(
            "resident graph capture must be ready for preparation."
        )
    compatibility = compare_resident_graph_capture_signature(
        lifecycle.signature, request
    )
    if not compatibility.compatible:
        raise ValueError("resident graph-capture signature is incompatible.")
    _validate_complete_resident_timestep_metadata(request, duration)
    compatibility = compare_resident_graph_capture_signature(
        lifecycle.signature, request
    )
    if not compatibility.compatible:
        raise ValueError("resident graph-capture signature is incompatible.")
    signature = lifecycle.signature
    return PreparedResidentTimestep(
        request=request,
        binding=binding,
        lifecycle=lifecycle,
        signature=signature,
        session=request.session,
        registry=request.registry,
        guard=request.guard,
        device=signature.device,
        dimensions=signature.dimensions,
        graph=request.graph,
        schedule=request.schedule,
        ordered_node_ids=request.schedule.ordered_node_ids,
        duration=duration,
        primary_arrays=signature.primary_arrays,
        resource_views=signature.resource_views,
    )
