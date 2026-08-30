"""Declare resident graph-capture capability and compatibility metadata.

This concrete, direct-import-only, declaration-only boundary resolves whether a
caller-provided probe reports graph-capture support and records identity-based
compatibility for an already-built resident request. It neither captures nor
replays graphs, imports Warp, probes devices itself, acquires resources,
launches work, transfers data, or synchronizes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, cast

from particula.execution import Backend, Device

if TYPE_CHECKING:
    from particula.execution.resident_scheduler import ResidentSimulationRequest
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )


class GraphCaptureAvailability(str, Enum):
    """Enumerate graph-capture availability outcomes from a lazy probe."""

    UNSUPPORTED_CPU = "unsupported_cpu"
    UNSUPPORTED_WARP_CPU = "unsupported_warp_cpu"
    UNAVAILABLE_RUNTIME = "unavailable_runtime"
    UNAVAILABLE_DEVICE = "unavailable_device"
    UNSUPPORTED_API = "unsupported_api"
    AVAILABLE = "available"


@dataclass(frozen=True, eq=False)
class GraphCaptureCapability:
    """Retain an exact device declaration and its capture availability.

    Attributes:
        device: Exact device declaration assessed by the resolver.
        availability: Availability outcome for ``device``.
    """

    device: Device
    availability: GraphCaptureAvailability

    def __post_init__(self) -> None:
        """Validate exact device and availability declaration types.

        Raises:
            TypeError: If either declaration does not have its required exact
                type.
        """
        if type(self.device) is not Device:
            raise TypeError("device must be an exact Device.")
        if type(self.availability) is not GraphCaptureAvailability:
            raise TypeError(
                "availability must be an exact GraphCaptureAvailability."
            )


class GraphCaptureRuntimeProbe(Protocol):
    """Declare caller-owned lazy runtime checks for graph-capture support.

    Implementations must return literal ``bool`` values. The resolver invokes
    methods only after the preceding availability condition succeeds.
    """

    def runtime_available(self) -> bool:
        """Return whether the optional runtime is available."""

    def device_available(self, device: Device) -> bool:
        """Return whether a declared device is available.

        Args:
            device: Exact device declaration to assess.
        """

    def capture_api_available(self, device: Device) -> bool:
        """Return whether a declared device exposes a capture API.

        Args:
            device: Exact device declaration to assess.
        """


def _require_probe_method(probe: object, name: str) -> Callable[..., object]:
    """Return one callable probe method or raise a deterministic error."""
    method = getattr(probe, name, None)
    if not callable(method):
        raise TypeError(f"probe.{name} must be callable.")
    return method


def _require_bool(result: object, name: str) -> bool:
    """Require a literal boolean result from a probe."""
    if type(result) is not bool:
        raise TypeError(f"probe.{name}() must return bool.")
    return result


def resolve_graph_capture_capability(
    device: Device, probe: GraphCaptureRuntimeProbe
) -> GraphCaptureCapability:
    """Resolve graph-capture capability without importing a runtime.

    CPU and Warp CPU declarations resolve without invoking ``probe``. Other
    devices invoke the caller-owned runtime, device, and capture-API checks in
    that order. Probe exceptions propagate unchanged.

    Args:
        device: Exact declared device to assess.
        probe: Caller-provided lazy runtime probe.

    Returns:
        Immutable capability declaration for ``device``.

    Raises:
        TypeError: If ``device`` is inexact, a probe method is not callable, or
            a probe does not return a literal ``bool``.
    """
    if type(device) is not Device:
        raise TypeError("device must be an exact Device.")
    if device.backend is Backend.CPU:
        return GraphCaptureCapability(
            device, GraphCaptureAvailability.UNSUPPORTED_CPU
        )
    if device.backend is Backend.WARP and device.native == "cpu":
        return GraphCaptureCapability(
            device, GraphCaptureAvailability.UNSUPPORTED_WARP_CPU
        )
    runtime_available = _require_probe_method(probe, "runtime_available")
    device_available = _require_probe_method(probe, "device_available")
    capture_api_available = _require_probe_method(
        probe, "capture_api_available"
    )
    if not _require_bool(runtime_available(), "runtime_available"):
        return GraphCaptureCapability(
            device, GraphCaptureAvailability.UNAVAILABLE_RUNTIME
        )
    if not _require_bool(device_available(device), "device_available"):
        return GraphCaptureCapability(
            device, GraphCaptureAvailability.UNAVAILABLE_DEVICE
        )
    if not _require_bool(
        capture_api_available(device), "capture_api_available"
    ):
        return GraphCaptureCapability(
            device, GraphCaptureAvailability.UNSUPPORTED_API
        )
    return GraphCaptureCapability(device, GraphCaptureAvailability.AVAILABLE)


class GraphCaptureDriftReason(str, Enum):
    """Enumerate signature groups in deterministic first-drift order."""

    REQUEST = "request"
    SESSION = "session"
    DEVICE = "device"
    DIMENSIONS = "dimensions"
    PRIMARY_CONTAINERS = "primary_containers"
    PRIMARY_ARRAYS = "primary_arrays"
    RESOURCE_VIEWS = "resource_views"
    GRAPH = "graph"
    SCHEDULE = "schedule"
    SCHEDULE_ORDER = "schedule_order"
    DIAGNOSTICS = "diagnostics"
    COMMUNICATION = "communication"
    CONFIGURATIONS = "configurations"
    RNG_RESOURCES = "rng_resources"


@dataclass(frozen=True, eq=False)
class GraphCaptureCompatibility:
    """State whether a request remains compatible with a signature.

    Attributes:
        compatible: Whether every tracked group retains object identity.
        reason: First changed group, or ``None`` only when compatible.
    """

    compatible: bool
    reason: GraphCaptureDriftReason | None

    def __post_init__(self) -> None:
        """Validate exact types and the compatible/reason invariant.

        Raises:
            TypeError: If a field does not have its required exact type.
            ValueError: If ``reason`` is absent or present inconsistently with
                ``compatible``.
        """
        if type(self.compatible) is not bool:
            raise TypeError("compatible must be bool.")
        if (
            self.reason is not None
            and type(self.reason) is not GraphCaptureDriftReason
        ):
            raise TypeError(
                "reason must be an exact GraphCaptureDriftReason or None."
            )
        if (self.compatible and self.reason is not None) or (
            not self.compatible and self.reason is None
        ):
            raise ValueError("reason must be None if and only if compatible.")


@dataclass(frozen=True, eq=False)
class ResidentGraphCaptureSignature:
    """Retain immutable identity metadata for one resident request.

    Every tuple holds references rather than payload values. The names mirror
    the ordered drift groups used by
    :func:`compare_resident_graph_capture_signature`.

    Attributes:
        request: Exact resident request instance.
        session: Request-bound resident session.
        device: Session metadata device declaration.
        dimensions: Session resident dimensions.
        primary_containers: Particle, gas, and environment containers.
        primary_arrays: Primary container arrays in schema order.
        resource_views: Request-owned published process and communication views.
        graph: Graph and its declaration tuples.
        schedule: Schedule and its declaration tuples.
        schedule_order: Resolved ordered schedule-node identifiers.
        diagnostics: Diagnostic plan, registrations, and output identities.
        communication: Optional communication declaration and view identities.
        configurations: Request-bound process configuration identities.
        rng_resources: Coagulation and wall-loss RNG array identities.
    """

    request: object
    session: object
    device: object
    dimensions: object
    primary_containers: tuple[object, ...]
    primary_arrays: tuple[object, ...]
    resource_views: tuple[object, ...]
    graph: tuple[object, ...]
    schedule: tuple[object, ...]
    schedule_order: tuple[object, ...]
    diagnostics: tuple[object, ...]
    communication: tuple[object, ...]
    configurations: tuple[object, ...]
    rng_resources: tuple[object, ...]

    def __post_init__(self) -> None:
        """Require exact immutable tuples for grouped identity metadata.

        Raises:
            TypeError: If a grouped metadata field is not an exact tuple.
        """
        for name in (
            "primary_containers",
            "primary_arrays",
            "resource_views",
            "graph",
            "schedule",
            "schedule_order",
            "diagnostics",
            "communication",
            "configurations",
            "rng_resources",
        ):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"{name} must be an exact tuple.")


def _resident_request_type() -> type[object]:
    """Lazily import the exact concrete request type after rejection checks."""
    from particula.execution.resident_scheduler import ResidentSimulationRequest

    return ResidentSimulationRequest


def _identity_tuple(*values: object) -> tuple[object, ...]:
    """Build a tuple whose members are compared by identity."""
    return values


def create_resident_graph_capture_signature(
    request: object,
) -> ResidentGraphCaptureSignature:
    """Create an identity-only structural signature for an exact request.

    The concrete request type is imported lazily after early inexact-request
    rejection. This function retains existing request metadata and published
    views only; it does not inspect array payloads or acquire resources.

    Args:
        request: Exact ``ResidentSimulationRequest`` to describe.

    Returns:
        Immutable identity metadata grouped in comparison precedence order.

    Raises:
        TypeError: If ``request`` is not an exact
            ``ResidentSimulationRequest``.
    """
    # Built-in values cannot be exact requests, so reject them before loading
    # Warp-dependent concrete request modules or touching arbitrary attributes.
    if (
        request is None
        or type(request) is object
        or type(request).__name__ != "ResidentSimulationRequest"
    ):
        raise TypeError("request must be an exact ResidentSimulationRequest.")
    request_type = _resident_request_type()
    if type(request) is not request_type:
        raise TypeError("request must be an exact ResidentSimulationRequest.")
    request = cast("ResidentSimulationRequest", request)

    session = request.session
    particles = cast("WarpParticleData", session.particles)
    gas = cast("WarpGasData", session.gas)
    environment = cast("WarpEnvironmentData", session.environment)
    condensation = request.condensation
    coagulation = request.coagulation
    wall_loss = request.wall_loss
    nucleation = request.nucleation
    communication = request.communication
    graph = request.graph
    schedule = request.schedule
    diagnostics = request.diagnostics
    communication_resources = (
        None if communication is None else communication.resources
    )
    return ResidentGraphCaptureSignature(
        request=request,
        session=session,
        device=session.metadata.device,
        dimensions=session.dimensions,
        primary_containers=_identity_tuple(particles, gas, environment),
        primary_arrays=_identity_tuple(
            particles.masses,
            particles.concentration,
            particles.density,
            particles.volume,
            particles.charge,
            gas.molar_mass,
            gas.concentration,
            gas.partitioning,
            gas.vapor_pressure,
            environment.temperature,
            environment.pressure,
            environment.saturation_ratio,
        ),
        resource_views=_identity_tuple(
            condensation.state.scratch_buffers,
            coagulation.resources,
            coagulation.resources.collision_pairs,
            coagulation.resources.n_collisions,
            wall_loss.resources,
            nucleation.resources,
            nucleation.resources.scratch,
            nucleation.resources.finalized_demand,
            nucleation.resources.diagnostics,
            nucleation.resources.exhaustion,
            communication_resources,
            (
                None
                if communication_resources is None
                else communication_resources.buffers
            ),
            (
                None
                if communication_resources is None
                else communication_resources.execution_state
            ),
            (
                None
                if communication_resources is None
                else communication_resources.final_volumes
            ),
        ),
        graph=_identity_tuple(graph, graph.nodes, graph.dependencies),
        schedule=_identity_tuple(
            schedule, schedule.nodes, schedule.dependencies
        ),
        schedule_order=_identity_tuple(schedule.ordered_node_ids),
        diagnostics=_identity_tuple(
            diagnostics,
            diagnostics.node,
            diagnostics.registrations,
            *(
                registration.operation
                for registration in diagnostics.registrations
            ),
            *(
                registration.output
                for registration in diagnostics.registrations
            ),
            *(
                item
                for registration in diagnostics.registrations
                for item in (
                    registration.energy_transfer,
                    registration.baseline_total_mass,
                    registration.source_ledger,
                    registration.sink_ledger,
                )
            ),
        ),
        communication=_identity_tuple(
            communication,
            None if communication is None else communication.communication_node,
            (
                None
                if communication is None
                else communication.volume_evolution_node
            ),
            (
                None
                if communication_resources is None
                else communication_resources.configuration
            ),
            (
                None
                if communication_resources is None
                else communication_resources.configuration.communication_map
            ),
            (
                None
                if communication_resources is None
                else communication_resources.buffers
            ),
            (
                None
                if communication_resources is None
                else communication_resources.execution_state
            ),
            (
                None
                if communication_resources is None
                else communication_resources.final_volumes
            ),
        ),
        configurations=_identity_tuple(
            request.registry,
            request.guard,
            request.thermodynamics,
            condensation,
            condensation.state,
            condensation.state.config,
            condensation.state.thermodynamics,
            coagulation,
            coagulation.request,
            coagulation.request.state,
            request.dilution,
            wall_loss,
            wall_loss.config,
            nucleation,
            nucleation.config,
            nucleation.exhaustion_controls,
            request.environment_update,
            request.gas_update,
        ),
        rng_resources=_identity_tuple(
            coagulation.resources.rng_states,
            wall_loss.resources.rng_states,
        ),
    )


def _same_identity(left: object, right: object) -> bool:
    """Compare arbitrary scalar or tuple signature entries by identity."""
    if type(left) is tuple and type(right) is tuple:
        return len(left) == len(right) and all(
            first is second for first, second in zip(left, right, strict=True)
        )
    return left is right


def compare_resident_graph_capture_signature(
    signature: ResidentGraphCaptureSignature, request: object
) -> GraphCaptureCompatibility:
    """Compare a request with a signature and return its first identity drift.

    A fresh metadata-only signature is created without recapturing, mutating,
    inspecting payload values, or replacing resident resources.

    Args:
        signature: Exact baseline signature to compare.
        request: Exact ``ResidentSimulationRequest`` to assess.

    Returns:
        Compatibility result containing the first changed group, if any.

    Raises:
        TypeError: If ``signature`` is inexact or ``request`` is not an exact
            ``ResidentSimulationRequest``.
    """
    if type(signature) is not ResidentGraphCaptureSignature:
        raise TypeError(
            "signature must be an exact ResidentGraphCaptureSignature."
        )
    current = create_resident_graph_capture_signature(request)
    for reason, field in (
        (GraphCaptureDriftReason.REQUEST, "request"),
        (GraphCaptureDriftReason.SESSION, "session"),
        (GraphCaptureDriftReason.DEVICE, "device"),
        (GraphCaptureDriftReason.DIMENSIONS, "dimensions"),
        (GraphCaptureDriftReason.PRIMARY_CONTAINERS, "primary_containers"),
        (GraphCaptureDriftReason.PRIMARY_ARRAYS, "primary_arrays"),
        (GraphCaptureDriftReason.RESOURCE_VIEWS, "resource_views"),
        (GraphCaptureDriftReason.GRAPH, "graph"),
        (GraphCaptureDriftReason.SCHEDULE, "schedule"),
        (GraphCaptureDriftReason.SCHEDULE_ORDER, "schedule_order"),
        (GraphCaptureDriftReason.DIAGNOSTICS, "diagnostics"),
        (GraphCaptureDriftReason.COMMUNICATION, "communication"),
        (GraphCaptureDriftReason.CONFIGURATIONS, "configurations"),
        (GraphCaptureDriftReason.RNG_RESOURCES, "rng_resources"),
    ):
        if not _same_identity(
            getattr(signature, field), getattr(current, field)
        ):
            return GraphCaptureCompatibility(False, reason)
    return GraphCaptureCompatibility(True, None)


class GraphCaptureLifecycleState(str, Enum):
    """Enumerate host-only graph-capture lifecycle metadata states."""

    READY = "ready"
    CAPTURED = "captured"
    INVALIDATED = "invalidated"
    FAULTED = "faulted"
    RETIRED = "retired"
    CLOSED = "closed"


class GraphCaptureFailureClassification(str, Enum):
    """Classify whether a host-recorded failure may follow a writer launch."""

    READ_ONLY = "read_only"
    WRITER_MAY_HAVE_LAUNCHED = "writer_may_have_launched"


@dataclass(frozen=True, eq=False)
class GraphCaptureLifecycle:
    """Retain immutable host metadata for one graph-capture lifecycle.

    This declaration records lifecycle intent only. It does not import Warp,
    allocate or capture a native graph, replay work, or act on a resident
    session.

    Attributes:
        capability: Exact graph-capture capability declaration retained by
            identity. Lifecycle creation requires it to be available.
        signature: Exact resident graph-capture identity signature.
        state: Current lifecycle declaration.
        first_invalidation_reason: First structural drift reason, if recorded.
    """

    capability: GraphCaptureCapability
    signature: ResidentGraphCaptureSignature
    state: GraphCaptureLifecycleState
    first_invalidation_reason: GraphCaptureDriftReason | None

    def __post_init__(self) -> None:
        """Validate lifecycle metadata types and reason-state invariants.

        Raises:
            TypeError: If a carrier, state, or invalidation reason has an
                inexact type.
            ValueError: If the invalidation reason is inconsistent with the
                lifecycle state.
        """
        if type(self.capability) is not GraphCaptureCapability:
            raise TypeError(
                "capability must be an exact GraphCaptureCapability."
            )
        if type(self.signature) is not ResidentGraphCaptureSignature:
            raise TypeError(
                "signature must be an exact ResidentGraphCaptureSignature."
            )
        if type(self.state) is not GraphCaptureLifecycleState:
            raise TypeError(
                "state must be an exact GraphCaptureLifecycleState."
            )
        if (
            self.first_invalidation_reason is not None
            and type(self.first_invalidation_reason)
            is not GraphCaptureDriftReason
        ):
            raise TypeError(
                "first_invalidation_reason must be an exact "
                "GraphCaptureDriftReason or None."
            )
        if (
            self.state
            in (
                GraphCaptureLifecycleState.READY,
                GraphCaptureLifecycleState.CAPTURED,
            )
            and self.first_invalidation_reason is not None
        ):
            raise ValueError("ready and captured lifecycles require no reason.")
        if (
            self.state is GraphCaptureLifecycleState.INVALIDATED
            and self.first_invalidation_reason is None
        ):
            raise ValueError("invalidated lifecycles require a reason.")


def _require_lifecycle(lifecycle: object) -> GraphCaptureLifecycle:
    """Require an exact lifecycle record before accessing its metadata."""
    if type(lifecycle) is not GraphCaptureLifecycle:
        raise TypeError("lifecycle must be an exact GraphCaptureLifecycle.")
    return lifecycle


def _lifecycle_successor(
    lifecycle: GraphCaptureLifecycle,
    state: GraphCaptureLifecycleState,
    reason: GraphCaptureDriftReason | None,
) -> GraphCaptureLifecycle:
    """Create a successor retaining P1 capability and signature identities."""
    return GraphCaptureLifecycle(
        capability=lifecycle.capability,
        signature=lifecycle.signature,
        state=state,
        first_invalidation_reason=reason,
    )


def create_graph_capture_lifecycle(
    capability: GraphCaptureCapability,
    signature: ResidentGraphCaptureSignature,
) -> GraphCaptureLifecycle:
    """Create ready host metadata without capturing or acting on a session.

    This declaration-only operation does not import Warp, allocate a graph, or
    validate or mutate a resident binding.

    Args:
        capability: Available graph-capture capability metadata.
        signature: Identity signature for the resident request.

    Returns:
        A new ready lifecycle record retaining both input carriers by identity.

    Raises:
        TypeError: If either P1 carrier is inexact.
        ValueError: If graph capture is not available.
    """
    if type(capability) is not GraphCaptureCapability:
        raise TypeError("capability must be an exact GraphCaptureCapability.")
    if type(signature) is not ResidentGraphCaptureSignature:
        raise TypeError(
            "signature must be an exact ResidentGraphCaptureSignature."
        )
    if capability.availability is not GraphCaptureAvailability.AVAILABLE:
        raise ValueError("capability must declare graph capture as available.")
    return GraphCaptureLifecycle(
        capability=capability,
        signature=signature,
        state=GraphCaptureLifecycleState.READY,
        first_invalidation_reason=None,
    )


def complete_graph_capture(
    lifecycle: GraphCaptureLifecycle,
) -> GraphCaptureLifecycle:
    """Declare capture completion without native or resident-session work.

    Args:
        lifecycle: Ready lifecycle metadata to transition.

    Returns:
        A captured successor retaining the capability and signature identities.

    Raises:
        TypeError: If ``lifecycle`` is not an exact lifecycle record.
        ValueError: If the lifecycle is not ready.
    """
    lifecycle = _require_lifecycle(lifecycle)
    if lifecycle.state is not GraphCaptureLifecycleState.READY:
        raise ValueError("graph capture can complete only from ready.")
    return _lifecycle_successor(
        lifecycle,
        GraphCaptureLifecycleState.CAPTURED,
        None,
    )


def invalidate_graph_capture(
    lifecycle: GraphCaptureLifecycle,
    compatibility: GraphCaptureCompatibility,
) -> GraphCaptureLifecycle:
    """Record structural invalidation host metadata without comparing requests.

    This operation neither imports Warp nor acts on a resident session.

    Args:
        lifecycle: Captured or already-invalidated lifecycle metadata.
        compatibility: P1 compatibility result containing the drift outcome.

    Returns:
        The original record for compatible or repeated invalidation, otherwise
        an invalidated successor retaining the first drift reason.

    Raises:
        TypeError: If either argument has an inexact type.
        ValueError: If the lifecycle is not captured or invalidated.
    """
    lifecycle = _require_lifecycle(lifecycle)
    if type(compatibility) is not GraphCaptureCompatibility:
        raise TypeError(
            "compatibility must be an exact GraphCaptureCompatibility."
        )
    if lifecycle.state not in (
        GraphCaptureLifecycleState.CAPTURED,
        GraphCaptureLifecycleState.INVALIDATED,
    ):
        raise ValueError("graph capture can invalidate only from captured.")
    if (
        compatibility.compatible
        or lifecycle.state is GraphCaptureLifecycleState.INVALIDATED
    ):
        return lifecycle
    return _lifecycle_successor(
        lifecycle,
        GraphCaptureLifecycleState.INVALIDATED,
        compatibility.reason,
    )


def classify_graph_capture_failure(
    lifecycle: GraphCaptureLifecycle,
    classification: GraphCaptureFailureClassification,
) -> GraphCaptureLifecycle:
    """Classify host failure metadata without executing a resident operation.

    This declaration-only operation does not import Warp or act on a resident
    session.

    Args:
        lifecycle: Lifecycle metadata associated with the failure.
        classification: Whether a writer may have launched before failure.

    Returns:
        The original record for read-only or faulted outcomes, or a faulted
        successor when a writer may have launched.

    Raises:
        TypeError: If either argument has an inexact type.
        ValueError: If the lifecycle is retired or closed.
    """
    lifecycle = _require_lifecycle(lifecycle)
    if type(classification) is not GraphCaptureFailureClassification:
        raise TypeError(
            "classification must be an exact GraphCaptureFailureClassification."
        )
    if lifecycle.state in (
        GraphCaptureLifecycleState.RETIRED,
        GraphCaptureLifecycleState.CLOSED,
    ):
        raise ValueError(
            "graph capture failure cannot be classified from terminal state."
        )
    if (
        classification is GraphCaptureFailureClassification.READ_ONLY
        or lifecycle.state is GraphCaptureLifecycleState.FAULTED
    ):
        return lifecycle
    return _lifecycle_successor(
        lifecycle,
        GraphCaptureLifecycleState.FAULTED,
        lifecycle.first_invalidation_reason,
    )


def retire_graph_capture(
    lifecycle: GraphCaptureLifecycle,
) -> GraphCaptureLifecycle:
    """Retire invalidated host metadata without native graph or session work.

    Args:
        lifecycle: Invalidated lifecycle metadata to retire.

    Returns:
        A retired successor, or the original retired record for repeated
        retirement.

    Raises:
        TypeError: If ``lifecycle`` is not an exact lifecycle record.
        ValueError: If the lifecycle is not invalidated or retired.
    """
    lifecycle = _require_lifecycle(lifecycle)
    if lifecycle.state is GraphCaptureLifecycleState.RETIRED:
        return lifecycle
    if lifecycle.state is not GraphCaptureLifecycleState.INVALIDATED:
        raise ValueError("graph capture can retire only from invalidated.")
    return _lifecycle_successor(
        lifecycle,
        GraphCaptureLifecycleState.RETIRED,
        lifecycle.first_invalidation_reason,
    )


def renew_retired_graph_capture(
    lifecycle: GraphCaptureLifecycle,
    signature: ResidentGraphCaptureSignature,
) -> GraphCaptureLifecycle:
    """Prepare ready host metadata without recapture or session work.

    This is the sole preparation path for a new declaration after retirement;
    it does not create a native graph or replace the predecessor.

    Args:
        lifecycle: Retired lifecycle metadata to renew.
        signature: New identity signature to retain in the ready record.

    Returns:
        A distinct ready lifecycle with the retired capability and new
        signature identities.

    Raises:
        TypeError: If either argument has an inexact type.
        ValueError: If the lifecycle is not retired.
    """
    lifecycle = _require_lifecycle(lifecycle)
    if type(signature) is not ResidentGraphCaptureSignature:
        raise TypeError(
            "signature must be an exact ResidentGraphCaptureSignature."
        )
    if lifecycle.state is not GraphCaptureLifecycleState.RETIRED:
        raise ValueError("graph capture can renew only from retired.")
    return GraphCaptureLifecycle(
        capability=lifecycle.capability,
        signature=signature,
        state=GraphCaptureLifecycleState.READY,
        first_invalidation_reason=None,
    )


def close_graph_capture(
    lifecycle: GraphCaptureLifecycle,
) -> GraphCaptureLifecycle:
    """Close host metadata without releasing native resources or a session.

    Args:
        lifecycle: Lifecycle metadata to close.

    Returns:
        A closed successor, or the original record when already closed.

    Raises:
        TypeError: If ``lifecycle`` is not an exact lifecycle record.
    """
    lifecycle = _require_lifecycle(lifecycle)
    if lifecycle.state is GraphCaptureLifecycleState.CLOSED:
        return lifecycle
    return _lifecycle_successor(
        lifecycle,
        GraphCaptureLifecycleState.CLOSED,
        lifecycle.first_invalidation_reason,
    )
