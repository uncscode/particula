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
