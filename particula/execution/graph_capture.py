"""Declare resident graph-capture capability and compatibility metadata.

This concrete, direct-import-only, declaration-only boundary resolves whether a
caller-provided probe reports graph-capture support and records identity-based
compatibility for an already-built resident request. It neither captures nor
replays graphs, imports Warp, probes devices itself, acquires resources,
launches work, transfers data, or synchronizes. Its binding helpers gate
scheduler admission and record explicit lifecycle successors without changing
resident payloads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, cast

from particula.execution import Backend, Device

if TYPE_CHECKING:
    from particula.execution.gpu_resources import GPUResourceRegistry
    from particula.execution.gpu_session import (
        ResidentSession,
        ResidentStepGuard,
    )
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


@dataclass(frozen=True, eq=False)
class GraphCaptureNativeCallables:
    """Retain native vocabulary without a native handle or cleanup callback."""

    capture_begin: Callable[..., object]
    capture_end: Callable[..., object]
    capture_instantiate: Callable[..., object]
    capture_launch: Callable[..., object]
    capture_release: Callable[..., object]

    def __post_init__(self) -> None:
        """Require every native vocabulary member to be callable."""
        for name in (
            "capture_begin",
            "capture_end",
            "capture_instantiate",
            "capture_launch",
            "capture_release",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable.")


class GraphCaptureRuntimeAdapter(GraphCaptureRuntimeProbe, Protocol):
    """Declare lazy runtime probes and callable resolution for P1."""

    def capture_callables(self, device: Device) -> GraphCaptureNativeCallables:
        """Return native callable vocabulary for one exact device."""


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
    if not _require_bool(runtime_available(), "runtime_available"):
        return GraphCaptureCapability(
            device, GraphCaptureAvailability.UNAVAILABLE_RUNTIME
        )
    device_available = _require_probe_method(probe, "device_available")
    if not _require_bool(device_available(device), "device_available"):
        return GraphCaptureCapability(
            device, GraphCaptureAvailability.UNAVAILABLE_DEVICE
        )
    capture_api_available = _require_probe_method(
        probe, "capture_api_available"
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
        configurations: Request-bound process configurations followed by the
            exact capture requirements, published resource set, and cached
            logical-byte report identities.
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


def _resident_session_type() -> type[object]:
    """Lazily return the exact resident-session carrier type."""
    from particula.execution.gpu_session import ResidentSession

    return ResidentSession


def _resident_guard_type() -> type[object]:
    """Lazily return the exact resident-step-guard carrier type."""
    from particula.execution.gpu_session import ResidentStepGuard

    return ResidentStepGuard


def _registry_type() -> type[object]:
    """Lazily return the exact pinned-resource-registry carrier type."""
    from particula.execution.gpu_resources import GPUResourceRegistry

    return GPUResourceRegistry


def _prepared_resident_simulation_type() -> type[object]:
    """Lazily return the exact prepared resident simulation carrier type."""
    from particula.execution.resident_scheduler import (
        PreparedResidentSimulation,
    )

    return PreparedResidentSimulation


def _prepared_resident_timestep_type() -> type[object]:
    """Lazily return the exact prepared resident timestep carrier type."""
    from particula.execution.resident_enqueue import PreparedResidentTimestep

    return PreparedResidentTimestep


def _capture_resource_requirements_type() -> type[object]:
    """Lazily return exact capture-resource requirements carrier type."""
    from particula.execution.gpu_resources import CaptureResourceRequirements

    return CaptureResourceRequirements


def _capture_resource_set_type() -> type[object]:
    """Lazily return exact published capture-resource set carrier type."""
    from particula.execution.gpu_resources import CaptureResourceSet

    return CaptureResourceSet


def _identity_tuple(*values: object) -> tuple[object, ...]:
    """Build a tuple whose members are compared by identity."""
    return values


def _nested_sidecar_identities(value: object) -> tuple[object, ...]:
    """Flatten dataclass sidecar leaves in deterministic declaration order."""
    fields = getattr(type(value), "__dataclass_fields__", None)
    if type(fields) is not dict:
        return (value,)
    identities: list[object] = []
    for name in fields:
        identities.extend(_nested_sidecar_identities(getattr(value, name)))
    return tuple(identities)


def validate_resident_capture_resources(request: object) -> object:
    """Resolve one published capture set with fixed identity checks only.

    This lazy concrete helper is deliberately metadata-only. It performs the
    registry's cached requirements lookup and confirms that the published set
    and cached immutable report still describe the request-bound resident
    resources. It does not prepare or acquire resources, rebuild accounting,
    inspect payloads, or synchronize.

    Args:
        request: Exact resident simulation request retaining published
            requirements.

    Returns:
        The exact registry-published capture resource set.

    Raises:
        TypeError: If ``request`` is not an exact resident simulation request.
        ValueError: If publication is absent, stale, or incompatible with the
            request-bound capture resources.
    """
    request_type = _resident_request_type()
    if type(request) is not request_type:
        raise TypeError("request must be an exact ResidentSimulationRequest.")
    typed = cast(Any, request)
    registry = typed.registry
    capture_set = cast(Any, registry).validate_capture_resource_set(
        typed.capture_resource_requirements
    )
    prepared = capture_set.prepared_views
    communication = typed.communication
    communication_resources = (
        None if communication is None else communication.resources
    )
    # Publication is an execution identity boundary: every final request
    # resource must be the exact object staged in the published capture set.
    if (
        capture_set.requirements is not typed.capture_resource_requirements
        or typed.capture_resource_requirements.session is not typed.session
        or capture_set.capacities
        is not typed.capture_resource_requirements.capacities
        or capture_set.inventory
        is not typed.capture_resource_requirements.inventory
        or capture_set.communication_resources
        is not typed.capture_resource_requirements.communication_resources
        or capture_set.communication_resources is not communication_resources
        or capture_set.condensation is not prepared.condensation
        or capture_set.coagulation is not prepared.coagulation
        or capture_set.wall_loss is not prepared.wall_loss
        or capture_set.nucleation is not prepared.nucleation
        or capture_set.dilution is not prepared.dilution
        or capture_set.condensation is None
        or capture_set.condensation.scratch_buffers
        is not typed.condensation.state.scratch_buffers
        or capture_set.coagulation is not typed.coagulation.resources
        or capture_set.wall_loss is not typed.wall_loss.resources
        or capture_set.nucleation is not typed.nucleation.resources
        or capture_set.dilution is None
        or capture_set.dilution is not typed.dilution.resources
        or capture_set.dilution.normalized_coefficient
        is not typed.dilution.coefficient
        or capture_set.inventory.registrations
        is not typed.diagnostics.registrations
        or capture_set.report is None
    ):
        raise ValueError("Capture resource set identities are incompatible.")
    return capture_set


def create_resident_graph_capture_signature(
    request: object,
    *,
    validate_capture_resources: bool = True,
    cached_capture_set: object | None = None,
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

    if validate_capture_resources:
        capture_set = validate_resident_capture_resources(request)
    else:
        # Signature comparison must diagnose earlier structural drift before a
        # stale registry binding is validated.  The retained capture set is
        # identity metadata only; the caller performs the authoritative
        # publication validation after ordered comparison succeeds.
        if cached_capture_set is None:
            raise ValueError("cached capture set is required for comparison.")
        capture_set = cached_capture_set

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
            *_nested_sidecar_identities(condensation.state.scratch_buffers),
            coagulation.resources,
            coagulation.resources.collision_pairs,
            coagulation.resources.n_collisions,
            wall_loss.resources,
            nucleation.resources,
            nucleation.resources.scratch,
            nucleation.resources.finalized_demand,
            nucleation.resources.diagnostics,
            nucleation.resources.exhaustion,
            *_nested_sidecar_identities(nucleation.resources.scratch),
            *_nested_sidecar_identities(nucleation.resources.finalized_demand),
            *_nested_sidecar_identities(nucleation.resources.diagnostics),
            *_nested_sidecar_identities(nucleation.resources.exhaustion),
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
            *(
                ()
                if communication_resources is None
                else _nested_sidecar_identities(communication_resources.buffers)
            ),
            *(
                ()
                if communication_resources is None
                else _nested_sidecar_identities(
                    communication_resources.execution_state
                )
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
            request.capture_resource_requirements,
            capture_set,
            cast(Any, capture_set).report,
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


def _compare_cached_admission_signature(  # noqa: C901
    signature: ResidentGraphCaptureSignature,
    request: object,
) -> GraphCaptureCompatibility:
    """Compare fixed admission identities without rebuilding a signature.

    The registry and publication gates separately validate pinned arrays and
    capture resources. This fast path checks the request-owned structural
    carriers that supported preparation retains by identity. The exhaustive
    public comparison remains the drift-diagnostic fallback.
    """
    if request is not signature.request:
        return GraphCaptureCompatibility(False, GraphCaptureDriftReason.REQUEST)
    typed = cast(Any, request)
    if typed.session is not signature.session:
        return GraphCaptureCompatibility(False, GraphCaptureDriftReason.SESSION)
    if typed.session.metadata.device is not signature.device:
        return GraphCaptureCompatibility(False, GraphCaptureDriftReason.DEVICE)
    if typed.session.dimensions is not signature.dimensions:
        return GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.DIMENSIONS
        )
    particles = typed.session.particles
    gas = typed.session.gas
    environment = typed.session.environment
    if (
        particles is not signature.primary_containers[0]
        or gas is not signature.primary_containers[1]
        or environment is not signature.primary_containers[2]
    ):
        return GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.PRIMARY_CONTAINERS
        )
    current_primary_arrays = _identity_tuple(
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
    )
    if not _same_identity(signature.primary_arrays, current_primary_arrays):
        return GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.PRIMARY_ARRAYS
        )
    expected_views = signature.resource_views
    view_index = 0

    def matches(value: object) -> bool:
        """Consume and compare one retained resource-view identity."""
        nonlocal view_index
        if (
            view_index >= len(expected_views)
            or value is not expected_views[view_index]
        ):
            return False
        view_index += 1
        return True

    def matches_nested(value: object) -> bool:
        """Compare dataclass leaves without constructing an identity tuple."""
        fields = getattr(type(value), "__dataclass_fields__", None)
        if type(fields) is not dict:
            return matches(value)
        return all(matches_nested(getattr(value, name)) for name in fields)

    condensation = typed.condensation
    coagulation = typed.coagulation
    wall_loss = typed.wall_loss
    nucleation = typed.nucleation
    communication = typed.communication
    communication_resources = (
        None if communication is None else communication.resources
    )
    if not (
        matches(condensation.state.scratch_buffers)
        and matches_nested(condensation.state.scratch_buffers)
        and matches(coagulation.resources)
        and matches(coagulation.resources.collision_pairs)
        and matches(coagulation.resources.n_collisions)
        and matches(wall_loss.resources)
        and matches(nucleation.resources)
        and matches(nucleation.resources.scratch)
        and matches(nucleation.resources.finalized_demand)
        and matches(nucleation.resources.diagnostics)
        and matches(nucleation.resources.exhaustion)
        and matches_nested(nucleation.resources.scratch)
        and matches_nested(nucleation.resources.finalized_demand)
        and matches_nested(nucleation.resources.diagnostics)
        and matches_nested(nucleation.resources.exhaustion)
        and matches(communication_resources)
        and matches(
            None
            if communication_resources is None
            else communication_resources.buffers
        )
        and matches(
            None
            if communication_resources is None
            else communication_resources.execution_state
        )
        and matches(
            None
            if communication_resources is None
            else communication_resources.final_volumes
        )
        and (
            communication_resources is None
            or (
                matches_nested(communication_resources.buffers)
                and matches_nested(communication_resources.execution_state)
            )
        )
        and view_index == len(expected_views)
    ):
        return GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.RESOURCE_VIEWS
        )
    if (
        typed.graph is not signature.graph[0]
        or typed.graph.nodes is not signature.graph[1]
        or typed.graph.dependencies is not signature.graph[2]
    ):
        return GraphCaptureCompatibility(False, GraphCaptureDriftReason.GRAPH)
    if (
        typed.schedule is not signature.schedule[0]
        or typed.schedule.nodes is not signature.schedule[1]
        or typed.schedule.dependencies is not signature.schedule[2]
    ):
        return GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.SCHEDULE
        )
    if typed.schedule.ordered_node_ids is not signature.schedule_order[0]:
        return GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.SCHEDULE_ORDER
        )
    diagnostics = typed.diagnostics
    current_diagnostics = _identity_tuple(
        diagnostics,
        diagnostics.node,
        diagnostics.registrations,
        *(item.operation for item in diagnostics.registrations),
        *(item.output for item in diagnostics.registrations),
        *(
            value
            for item in diagnostics.registrations
            for value in (
                item.energy_transfer,
                item.baseline_total_mass,
                item.source_ledger,
                item.sink_ledger,
            )
        ),
    )
    if not _same_identity(signature.diagnostics, current_diagnostics):
        return GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.DIAGNOSTICS
        )
    current_communication = _identity_tuple(
        communication,
        None if communication is None else communication.communication_node,
        None if communication is None else communication.volume_evolution_node,
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
    )
    if not _same_identity(signature.communication, current_communication):
        return GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.COMMUNICATION
        )
    configurations = signature.configurations
    capture_set = configurations[-2]
    current_configurations = _identity_tuple(
        typed.registry,
        typed.guard,
        typed.thermodynamics,
        condensation,
        condensation.state,
        condensation.state.config,
        condensation.state.thermodynamics,
        coagulation,
        coagulation.request,
        coagulation.request.state,
        typed.dilution,
        wall_loss,
        wall_loss.config,
        nucleation,
        nucleation.config,
        nucleation.exhaustion_controls,
        typed.environment_update,
        typed.gas_update,
        typed.capture_resource_requirements,
        capture_set,
        cast(Any, capture_set).report,
    )
    if not _same_identity(configurations, current_configurations):
        return GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.CONFIGURATIONS
        )
    current_rng = _identity_tuple(
        coagulation.resources.rng_states,
        wall_loss.resources.rng_states,
    )
    if not _same_identity(signature.rng_resources, current_rng):
        return GraphCaptureCompatibility(
            False, GraphCaptureDriftReason.RNG_RESOURCES
        )
    return GraphCaptureCompatibility(True, None)


def compare_resident_graph_capture_signature(
    signature: ResidentGraphCaptureSignature,
    request: object,
    *,
    admission_token: ResidentGraphCaptureSignature | None = None,
) -> GraphCaptureCompatibility:
    """Compare a request with a signature and return its first identity drift.

    By default, a fresh metadata-only signature is created for complete ordered
    drift diagnosis. Admission callers that already validated their exact
    frozen binding may pass that retained signature itself as
    ``admission_token``. The identity token provides the compatible fast path
    without rebuilding signature tuples; any different token fails closed.

    Args:
        signature: Exact baseline signature to compare.
        request: Exact ``ResidentSimulationRequest`` to assess.
        admission_token: Optional exact retained signature authorizing the
            cached compatible-admission path.

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
    if admission_token is not None:
        if type(admission_token) is not ResidentGraphCaptureSignature:
            raise TypeError(
                "admission_token must be an exact "
                "ResidentGraphCaptureSignature or None."
            )
        if admission_token is not signature:
            raise ValueError("admission_token must be the retained signature.")
        return _compare_cached_admission_signature(signature, request)
    current = create_resident_graph_capture_signature(
        request,
        validate_capture_resources=False,
        cached_capture_set=signature.configurations[-2],
    )
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
            self.capability.availability
            is not GraphCaptureAvailability.AVAILABLE
        ):
            raise ValueError("lifecycle capability must be available.")
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
            self.state
            in (
                GraphCaptureLifecycleState.INVALIDATED,
                GraphCaptureLifecycleState.RETIRED,
            )
            and self.first_invalidation_reason is None
        ):
            raise ValueError(
                "invalidated and retired lifecycles require a reason."
            )


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


@dataclass(eq=False)
class ResidentGraphCaptureBinding:
    """Retain one exact resident binding and its mutable lifecycle metadata.

    This direct-module-only carrier owns lifecycle successors for the attached
    final request. It is metadata only: it neither captures nor replays a graph,
    accesses resident payloads, transfers data, synchronizes, or falls back.

    Attributes:
        lifecycle: Current immutable lifecycle metadata retained by the binding.
    """

    _request: object
    _session: object
    _registry: object
    _guard: object
    _lifecycle: GraphCaptureLifecycle

    def __post_init__(self) -> None:
        """Require the exact, mutually identity-bound resident carriers."""
        _validate_resident_binding(
            self._request,
            self._session,
            self._registry,
            self._guard,
            self._lifecycle,
        )

    @property
    def lifecycle(self) -> GraphCaptureLifecycle:
        """Return the lifecycle currently owned by this binding."""
        return self._lifecycle


def _require_exact_resident_carrier(
    value: object,
    name: str,
    expected_name: str,
    resolver: Callable[[], type[object]],
) -> object:
    """Reject obvious inexact input before lazily importing resident modules."""
    if value is None or type(value).__name__ != expected_name:
        raise TypeError(f"{name} must be an exact {expected_name}.")
    if type(value) is not resolver():
        raise TypeError(f"{name} must be an exact {expected_name}.")
    return value


def _validate_resident_binding(
    request: object,
    session: object,
    registry: object,
    guard: object,
    lifecycle: object,
) -> None:
    """Validate retained identity links without reading resident payloads."""
    request = cast(
        "ResidentSimulationRequest",
        _require_exact_resident_carrier(
            request,
            "request",
            "ResidentSimulationRequest",
            _resident_request_type,
        ),
    )
    session = cast(
        "ResidentSession",
        _require_exact_resident_carrier(
            session, "session", "ResidentSession", _resident_session_type
        ),
    )
    registry = cast(
        "GPUResourceRegistry",
        _require_exact_resident_carrier(
            registry, "registry", "GPUResourceRegistry", _registry_type
        ),
    )
    guard = cast(
        "ResidentStepGuard",
        _require_exact_resident_carrier(
            guard, "guard", "ResidentStepGuard", _resident_guard_type
        ),
    )
    lifecycle = _require_lifecycle(lifecycle)
    if (
        request.session is not session
        or request.registry is not registry
        or request.guard is not guard
        or guard._session is not session
        or guard._registry is not registry
        or registry._session is not session
        or lifecycle.signature.request is not request
        or lifecycle.signature.session is not session
    ):
        raise ValueError(
            "resident graph-capture binding identities do not match."
        )


def _require_binding(binding: object) -> ResidentGraphCaptureBinding:
    """Require an exact graph-capture binding before reading its fields."""
    if type(binding) is not ResidentGraphCaptureBinding:
        raise TypeError("binding must be an exact ResidentGraphCaptureBinding.")
    return binding


def _require_attached_resident_binding(
    binding: object,
) -> ResidentGraphCaptureBinding:
    """Require an exact binding attached to its retained final request."""
    binding = _require_binding(binding)
    _validate_resident_binding(
        binding._request,
        binding._session,
        binding._registry,
        binding._guard,
        binding._lifecycle,
    )
    request = cast("ResidentSimulationRequest", binding._request)
    if request.graph_capture_binding is not binding:
        raise ValueError("request graph-capture attachment does not match.")
    return binding


def _attach_resident_graph_capture_binding(
    request: object, binding: object
) -> None:
    """Attach one exact binding to a final frozen request exactly once.

    The request must be constructed without an attachment first. After all
    retained identities are checked, this construction-only helper performs the
    sole assignment to the frozen request's optional binding field.

    Args:
        request: Exact resident simulation request to attach.
        binding: Exact binding retaining the same request and resident carriers.

    Raises:
        TypeError: If either argument is not the required exact concrete type.
        ValueError: If the binding retains another request or the request is
            already attached.
    """
    request = cast(
        "ResidentSimulationRequest",
        _require_exact_resident_carrier(
            request,
            "request",
            "ResidentSimulationRequest",
            _resident_request_type,
        ),
    )
    binding = _require_binding(binding)
    _validate_resident_binding(
        binding._request,
        binding._session,
        binding._registry,
        binding._guard,
        binding._lifecycle,
    )
    if binding._request is not request:
        raise ValueError("binding must retain the exact request.")
    if request.graph_capture_binding is not None:
        raise ValueError("request already has a graph-capture binding.")
    object.__setattr__(request, "graph_capture_binding", binding)


def complete_resident_graph_capture(binding: object) -> GraphCaptureLifecycle:
    """Explicitly declare capture completion on one retained binding.

    This metadata transition does not perform native graph capture or resident
    work.

    Args:
        binding: Exact resident graph-capture binding to transition.

    Returns:
        Captured lifecycle metadata now owned by ``binding``.

    Raises:
        TypeError: If ``binding`` is not an exact binding.
        ValueError: If the retained carriers are inconsistent or the lifecycle
            is not ready.
    """
    binding = _require_attached_resident_binding(binding)
    binding._lifecycle = complete_graph_capture(binding._lifecycle)
    return binding._lifecycle


@dataclass(frozen=True, eq=False)
class PreparedGraphCaptureQualification:
    """Retain one READY qualification without a native handle or cleanup.

    P1 retains the exact prepared resident binding, published capture resources,
    and native callable vocabulary by identity. It creates no graph/exec handle
    and owns no cleanup callback. Successful qualification preserves READY;
    P2/P3 alone own native capture, handles, release, and cleanup.
    """

    binding: ResidentGraphCaptureBinding
    lifecycle: GraphCaptureLifecycle
    signature: ResidentGraphCaptureSignature
    request: object
    session: object
    registry: object
    guard: object
    prepared: object
    timestep: object
    capture_requirements: object
    capture_set: object
    capture_report: object
    device: Device
    dimensions: object
    graph: object
    schedule: object
    ordered_node_ids: tuple[object, ...]
    duration: object
    duration_is_identity: bool
    primary_arrays: tuple[object, ...]
    resource_views: tuple[object, ...]
    native_callables: GraphCaptureNativeCallables


def _require_adapter_method(
    adapter: object, name: str
) -> Callable[..., object]:
    """Return one callable adapter method without invoking it."""
    method = getattr(adapter, name, None)
    if not callable(method):
        raise TypeError(f"adapter.{name} must be callable.")
    return method


def _raise_unavailable_qualification(
    capability: GraphCaptureCapability,
) -> None:
    """Map a resolved unavailable capability to a deterministic error."""
    messages = {
        GraphCaptureAvailability.UNSUPPORTED_CPU: (
            "graph capture requires a CUDA resident device."
        ),
        GraphCaptureAvailability.UNSUPPORTED_WARP_CPU: (
            "graph capture requires a CUDA resident device."
        ),
        GraphCaptureAvailability.UNAVAILABLE_RUNTIME: (
            "graph capture runtime is unavailable."
        ),
        GraphCaptureAvailability.UNAVAILABLE_DEVICE: (
            "graph capture device is unavailable."
        ),
        GraphCaptureAvailability.UNSUPPORTED_API: (
            "graph capture API is unsupported."
        ),
    }
    if capability.availability is not GraphCaptureAvailability.AVAILABLE:
        raise ValueError(messages[capability.availability])


def qualify_prepared_resident_graph_capture(  # noqa: C901
    binding: object,
    prepared: object,
    capture_set: object,
    adapter: GraphCaptureRuntimeAdapter,
) -> PreparedGraphCaptureQualification:
    """Qualify one prepared READY binding for a later native capture phase.

    This metadata-only P1 boundary validates existing exact identities, resolves
    the adapter vocabulary lazily, and returns a fresh frozen record. It does
    not open a guard token, invoke native callables, capture, enqueue, allocate,
    synchronize, transfer, mutate lifecycle state, or release native resources.

    Args:
        binding: Exact attached READY resident graph-capture binding.
        prepared: Exact E8-F2 prepared resident simulation.
        capture_set: Exact E8-F3 published capture resource set.
        adapter: Caller-owned lazy native runtime adapter.

    Returns:
        A fresh qualification retaining only existing metadata identities.

    Raises:
        TypeError: If a carrier, adapter member, probe result, or callable
            record has an invalid exact type.
        ValueError: If READY binding metadata or capability qualification fails.
    """
    binding = _require_attached_resident_binding(binding)
    prepared = _require_exact_resident_carrier(
        prepared,
        "prepared",
        "PreparedResidentSimulation",
        _prepared_resident_simulation_type,
    )
    capture_set = _require_exact_resident_carrier(
        capture_set,
        "capture_set",
        "CaptureResourceSet",
        _capture_resource_set_type,
    )
    prepared_any = cast(Any, prepared)
    capture_set_any = cast(Any, capture_set)
    request = cast("ResidentSimulationRequest", binding._request)
    session = cast("ResidentSession", binding._session)
    registry = cast("GPUResourceRegistry", binding._registry)
    guard = cast("ResidentStepGuard", binding._guard)
    lifecycle = binding._lifecycle
    signature = lifecycle.signature
    timestep = prepared_any.timestep
    if type(timestep) is not _prepared_resident_timestep_type():
        raise TypeError("timestep must be an exact PreparedResidentTimestep.")
    requirements = prepared_any.capture_requirements
    if type(requirements) is not _capture_resource_requirements_type():
        raise TypeError(
            "capture_requirements must be an exact CaptureResourceRequirements."
        )
    if (
        prepared_any.request is not request
        or prepared_any.session is not session
        or prepared_any.registry is not registry
        or prepared_any.guard is not guard
        or prepared_any.lifecycle is not lifecycle
        or prepared_any.signature is not signature
        or timestep.request is not request
        or timestep.binding is not binding
        or timestep.lifecycle is not lifecycle
        or timestep.signature is not signature
        or timestep.session is not session
        or timestep.registry is not registry
        or timestep.guard is not guard
        or prepared_any.graph is not request.graph
        or prepared_any.schedule is not request.schedule
        or prepared_any.ordered_node_ids
        is not request.schedule.ordered_node_ids
        or prepared_any.primary_arrays is not signature.primary_arrays
        or prepared_any.resource_views is not signature.resource_views
        or prepared_any.capture_requirements is not requirements
        or requirements is not request.capture_resource_requirements
        or prepared_any.capture_set is not capture_set
        or timestep.capture_set is not capture_set
        or prepared_any.capture_report is not capture_set_any.report
        or timestep.capture_report is not capture_set_any.report
        or capture_set_any.requirements is not requirements
        or signature.configurations[-3] is not requirements
        or signature.configurations[-2] is not capture_set
        or signature.configurations[-1] is not capture_set_any.report
    ):
        raise ValueError("prepared graph-capture identities do not match.")
    duration_is_identity = prepared_any.duration is timestep.duration
    if not duration_is_identity and prepared_any.duration != timestep.duration:
        raise ValueError("prepared graph-capture durations do not match.")
    guard.assert_step_closed()
    registry.assert_step_closed()
    registry.validate_pinned_session(session)
    if session.lifecycle.name != "ACTIVE":
        raise ValueError("resident session must be ACTIVE.")
    capability = lifecycle.capability
    if capability.device != session.metadata.device:
        raise ValueError(
            "graph-capture capability device does not match session."
        )
    if capability.availability is not GraphCaptureAvailability.AVAILABLE:
        raise ValueError("graph capture capability must be available.")
    if (
        capability.device.backend is not Backend.WARP
        or capability.device.native == "cpu"
    ):
        raise ValueError("graph capture requires a CUDA resident device.")
    if lifecycle.state is not GraphCaptureLifecycleState.READY:
        raise ValueError("graph capture lifecycle must be ready.")
    compatibility = compare_resident_graph_capture_signature(
        signature,
        request,
        admission_token=signature,
    )
    if not compatibility.compatible:
        raise ValueError("resident graph-capture signature is incompatible.")
    if validate_resident_capture_resources(request) is not capture_set:
        raise ValueError("prepared capture resource set does not match.")
    resolved = resolve_graph_capture_capability(
        session.metadata.device, adapter
    )
    _raise_unavailable_qualification(resolved)
    callables = _require_adapter_method(adapter, "capture_callables")(
        session.metadata.device
    )
    if type(callables) is not GraphCaptureNativeCallables:
        raise TypeError(
            "adapter.capture_callables() must return an exact "
            "GraphCaptureNativeCallables."
        )
    return PreparedGraphCaptureQualification(
        binding=binding,
        lifecycle=lifecycle,
        signature=signature,
        request=request,
        session=session,
        registry=registry,
        guard=guard,
        prepared=prepared,
        timestep=timestep,
        capture_requirements=requirements,
        capture_set=capture_set,
        capture_report=capture_set_any.report,
        device=session.metadata.device,
        dimensions=session.dimensions,
        graph=request.graph,
        schedule=request.schedule,
        ordered_node_ids=request.schedule.ordered_node_ids,
        duration=prepared_any.duration,
        duration_is_identity=duration_is_identity,
        primary_arrays=signature.primary_arrays,
        resource_views=signature.resource_views,
        native_callables=callables,
    )


def gate_resident_graph_capture(binding: object) -> None:
    """Fail closed unless a captured binding remains exactly dispatchable.

    The gate verifies the attached request, resident binding, closed guard,
    active pinned session, available CUDA capability, captured lifecycle, and
    unchanged structural signature before scheduler token entry. Structural
    drift invalidates a captured lifecycle; all other rejection is read-only.
    This operation neither captures nor replays graphs, transfers data,
    synchronizes, acquires resources, or falls back.

    Args:
        binding: Exact resident graph-capture binding to admit.

    Raises:
        TypeError: If ``binding`` or its retained carriers have inexact types.
        ValueError: If the binding is stale, unavailable, not captured, or
            structurally incompatible with its resident request.
    """
    binding = _require_attached_resident_binding(binding)
    request = cast("ResidentSimulationRequest", binding._request)
    session = cast("ResidentSession", binding._session)
    registry = cast("GPUResourceRegistry", binding._registry)
    guard = cast("ResidentStepGuard", binding._guard)
    if request.graph_capture_binding is not binding:
        raise ValueError("request graph-capture attachment does not match.")
    guard.assert_step_closed()
    registry.validate_pinned_session(session)
    if session.lifecycle.name != "ACTIVE":
        raise ValueError("resident session must be ACTIVE.")
    capability = binding._lifecycle.capability
    if capability.device != session.metadata.device:
        raise ValueError(
            "graph-capture capability device does not match session."
        )
    if capability.availability is not GraphCaptureAvailability.AVAILABLE:
        raise ValueError("graph capture capability must be available.")
    if (
        capability.device.backend is not Backend.WARP
        or capability.device.native == "cpu"
    ):
        raise ValueError("graph capture requires a CUDA resident device.")
    if binding._lifecycle.state is not GraphCaptureLifecycleState.CAPTURED:
        raise ValueError("graph capture lifecycle must be captured.")
    # Requirements are part of the ordered ``configurations`` signature group.
    # Compare that retained identity before the current publication lookup so a
    # changed valid publication invalidates CAPTURED metadata as configuration
    # drift instead of being reported as an early lookup error.
    if (
        binding._lifecycle.signature.configurations[-3]
        is not request.capture_resource_requirements
    ):
        binding._lifecycle = invalidate_graph_capture(
            binding._lifecycle,
            GraphCaptureCompatibility(
                False, GraphCaptureDriftReason.CONFIGURATIONS
            ),
        )
        raise ValueError("resident graph-capture signature is incompatible.")
    compatibility = compare_resident_graph_capture_signature(
        binding._lifecycle.signature,
        request,
        admission_token=binding._lifecycle.signature,
    )
    if not compatibility.compatible:
        binding._lifecycle = invalidate_graph_capture(
            binding._lifecycle, compatibility
        )
        raise ValueError("resident graph-capture signature is incompatible.")
    validate_resident_capture_resources(request)


def classify_resident_graph_capture_writer_failure(binding: object) -> None:
    """Record a scheduler-confirmed possible writer failure on a binding.

    The scheduler calls this only after its existing cleanup determines that a
    writer may have launched. It records metadata only and does not retry,
    roll back, or act on resident resources.

    Args:
        binding: Exact binding whose lifecycle receives the classification.

    Raises:
        TypeError: If ``binding`` is not an exact binding.
        ValueError: If its lifecycle is retired or closed.
    """
    binding = _require_attached_resident_binding(binding)
    binding._lifecycle = classify_graph_capture_failure(
        binding._lifecycle,
        GraphCaptureFailureClassification.WRITER_MAY_HAVE_LAUNCHED,
    )


def _fault_resident_graph_capture_after_classification_failure(
    binding: object,
) -> None:
    """Fault an attached binding after classification itself fails."""
    binding = _require_attached_resident_binding(binding)
    if binding._lifecycle.state in (
        GraphCaptureLifecycleState.RETIRED,
        GraphCaptureLifecycleState.CLOSED,
    ):
        raise ValueError("terminal graph capture cannot be faulted.")
    binding._lifecycle = _lifecycle_successor(
        binding._lifecycle,
        GraphCaptureLifecycleState.FAULTED,
        binding._lifecycle.first_invalidation_reason,
    )


def retire_resident_graph_capture(binding: object) -> GraphCaptureLifecycle:
    """Explicitly retire invalidated lifecycle metadata on one binding.

    Args:
        binding: Exact binding that owns invalidated lifecycle metadata.

    Returns:
        Retired lifecycle metadata now owned by ``binding``.

    Raises:
        TypeError: If ``binding`` is not an exact binding.
        ValueError: If the lifecycle is neither invalidated nor already
            retired.
    """
    binding = _require_attached_resident_binding(binding)
    binding._lifecycle = retire_graph_capture(binding._lifecycle)
    return binding._lifecycle


def renew_resident_graph_capture(
    binding: object, signature: object
) -> GraphCaptureLifecycle:
    """Explicitly renew a retired exact binding with a new request signature.

    Renewal prepares ready metadata only; callers must separately declare
    capture completion before the scheduler can admit the binding. It does not
    recapture or replay a graph, replace resources, transfer data, synchronize,
    or fall back.

    Args:
        binding: Exact binding that owns retired lifecycle metadata.
        signature: Exact new signature for the binding's retained request.

    Returns:
        Ready lifecycle metadata now owned by ``binding``.

    Raises:
        TypeError: If ``binding`` or ``signature`` is inexact.
        ValueError: If retained identities differ or the lifecycle is not
            retired.
    """
    binding = _require_attached_resident_binding(binding)
    if type(signature) is not ResidentGraphCaptureSignature:
        raise TypeError(
            "signature must be an exact ResidentGraphCaptureSignature."
        )
    if (
        signature.request is not binding._request
        or signature.session is not binding._session
    ):
        raise ValueError("renewal signature must retain the exact binding.")
    binding._lifecycle = renew_retired_graph_capture(
        binding._lifecycle, signature
    )
    return binding._lifecycle


def close_resident_graph_capture(binding: object) -> GraphCaptureLifecycle:
    """Close lifecycle metadata owned by one exact attached binding.

    Repeated closure returns the identical closed lifecycle. This metadata-only
    operation does not dispatch, release resources, or mutate resident state.

    Args:
        binding: Exact attached resident graph-capture binding to close.

    Returns:
        Closed lifecycle metadata now owned by ``binding``.

    Raises:
        TypeError: If ``binding`` is not an exact binding.
        ValueError: If retained ownership or attachment identities are stale.
    """
    binding = _require_attached_resident_binding(binding)
    binding._lifecycle = close_graph_capture(binding._lifecycle)
    return binding._lifecycle
