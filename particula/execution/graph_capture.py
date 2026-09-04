"""Qualify and capture prepared GPU-resident timestep graphs.

This concrete, direct-import-only boundary resolves caller-provided
graph-capture support and records identity-based compatibility for an
already-built resident request. Its prepared-qualification boundary lazily
obtains an adapter's callable vocabulary for one exact READY binding. Its
capture boundary calls native begin, retained-operation dispatch, and end in
order, retains the resulting opaque handle, and privately releases it when
post-capture validation fails. It neither instantiates, launches, nor replays
graphs; imports Warp; probes devices itself; acquires resources; transfers data;
or synchronizes. Its binding helpers gate scheduler admission and record
explicit lifecycle successors without changing resident payloads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, cast

from particula.execution import Backend, Device

_ACTIVE_CAPTURE_BINDING_IDS: set[int] = set()
_ACTIVE_CAPTURE_LOCK = Lock()

if TYPE_CHECKING:
    from particula.execution.gpu_resources import (
        CaptureResourceSet,
        GPUResourceRegistry,
    )
    from particula.execution.gpu_session import (
        ResidentSession,
        ResidentStepGuard,
    )
    from particula.execution.resident_enqueue import PreparedResidentTimestep
    from particula.execution.resident_scheduler import (
        PreparedResidentSimulation,
        ResidentSimulationRequest,
    )
    from particula.execution.scheduler import ResolvedTimestepSchedule
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
        """Return whether the optional graph-capture runtime is available.

        Returns:
            ``True`` when the runtime can be queried for graph-capture support.
        """

    def device_available(self, device: Device) -> bool:
        """Return whether a declared device is available.

        Args:
            device: Exact device declaration to assess.

        Returns:
            ``True`` when the declared device is available to the runtime.
        """

    def capture_api_available(self, device: Device) -> bool:
        """Return whether a declared device exposes a capture API.

        Args:
            device: Exact device declaration to assess.

        Returns:
            ``True`` when the device exposes the required capture API.
        """


@dataclass(frozen=True, eq=False)
class GraphCaptureNativeCallables:
    """Retain an adapter's native callable vocabulary by exact identity.

    This record is vocabulary only: it holds no graph or executable handle and
    owns no cleanup callback. P2 invokes only ``capture_begin()``,
    ``capture_end()``, and, after a post-end rejection, ``capture_release()``.
    It never invokes ``capture_instantiate`` or ``capture_launch``. In
    particular, ``capture_release`` is a callable for a later owner to invoke,
    not a retained cleanup action.

    Attributes:
        capture_begin: Callable that begins native graph capture.
        capture_end: Callable that ends native graph capture.
        capture_instantiate: Callable that instantiates a captured graph.
        capture_launch: Callable that launches an instantiated graph.
        capture_release: Callable that releases a native graph resource.
    """

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
    """Declare lazy runtime probes and callable resolution for P1.

    The adapter owns runtime-specific behavior and treats ``Device.native`` as
    opaque. Qualification resolves runtime, device, and API availability in
    order, then calls ``capture_callables`` once only after all checks pass. It
    retains the returned exact vocabulary by identity and never invokes it.
    """

    def capture_callables(self, device: Device) -> GraphCaptureNativeCallables:
        """Return native callable vocabulary for one exactly qualified device.

        Args:
            device: Exact device declaration whose capture API was qualified.

        Returns:
            Exact callable vocabulary retained for a later native capture
            phase; this call creates neither a native handle nor cleanup work.
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
        validate_capture_resources: Whether to validate the published resource
            set before creating the signature.
        cached_capture_set: Previously validated resource set used when
            ``validate_capture_resources`` is ``False``.

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
    """Retain one exact READY qualification without a native handle or cleanup.

    P1 retains the prepared resident binding, published capture resources, and
    native callable vocabulary by identity; it copies no payloads. It creates
    no graph or executable handle, owns no cleanup callback, and does not
    invoke any retained callable. Successful qualification leaves the binding's
    lifecycle in READY; P2/P3 alone own native capture, handles, release, and
    cleanup.

    Attributes:
        binding: Attached resident graph-capture binding being qualified.
        lifecycle: READY lifecycle retained by the binding.
        signature: Structural identity signature for the request.
        request: Exact resident simulation request.
        session: Exact resident session attached to the request.
        registry: Exact resource registry pinned to the session.
        guard: Exact closed step guard attached to the session.
        prepared: Exact prepared resident simulation.
        timestep: Exact prepared timestep metadata.
        capture_requirements: Published capture-resource requirements.
        capture_set: Published capture-resource set.
        capture_report: Cached logical-byte report for the capture set.
        device: Exact non-CPU Warp device qualified for capture.
        dimensions: Resident dimensions retained by the session.
        graph: Prepared graph declaration.
        schedule: Prepared schedule declaration.
        ordered_node_ids: Ordered schedule-node identifiers.
        duration: Prepared timestep duration.
        duration_is_identity: Whether prepared and timestep durations are the
            same object rather than merely equal values.
        primary_arrays: Identity tuple of resident primary arrays.
        resource_views: Identity tuple of resident resource views.
        native_callables: Adapter-provided callable vocabulary for later use.
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


@dataclass(frozen=True, eq=False)
class CapturedResidentGraph:
    """Retain one completed native capture and its exact resident binding.

    Authentic records are issued only after P2 capture completes and publishes
    CAPTURED metadata. The opaque ``handle`` is retained by identity only; its
    type, equality, truthiness, and serialization are never inspected. P2 owns
    native begin/end/release work; P3 may only forward an authentic handle to
    ``capture_launch``. This concrete carrier neither instantiates nor launches
    the graph and does not expose cleanup.

    Attributes:
        qualification: Exact READY qualification validated before and after
            native capture.
        binding: Exact attached resident graph-capture binding.
        lifecycle: Binding-owned CAPTURED lifecycle successor.
        signature: Exact structural signature retained by ``qualification``.
        request: Exact resident simulation request.
        session: Exact active resident session.
        registry: Exact resource registry pinned to ``session``.
        guard: Exact closed resident step guard.
        prepared: Exact prepared resident simulation dispatched during capture.
        timestep: Exact prepared timestep retained by ``prepared``.
        capture_requirements: Exact published capture-resource requirements.
        capture_set: Exact published capture-resource set.
        capture_report: Exact cached logical-byte report for ``capture_set``.
        device: Exact qualified non-CPU Warp device.
        handle: Opaque handle returned by native ``capture_end()``.
    """

    qualification: PreparedGraphCaptureQualification
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
    handle: object

    def __post_init__(self) -> None:
        """Require the exact captured successor and qualification identities."""
        if type(self.qualification) is not PreparedGraphCaptureQualification:
            raise TypeError(
                "qualification must be an exact "
                "PreparedGraphCaptureQualification."
            )
        qualification = self.qualification
        if (
            self.binding is not qualification.binding
            or self.lifecycle is not self.binding.lifecycle
            or self.lifecycle.state is not GraphCaptureLifecycleState.CAPTURED
            or self.signature is not qualification.signature
            or self.request is not qualification.request
            or self.session is not qualification.session
            or self.registry is not qualification.registry
            or self.guard is not qualification.guard
            or self.prepared is not qualification.prepared
            or self.timestep is not qualification.timestep
            or self.capture_requirements
            is not qualification.capture_requirements
            or self.capture_set is not qualification.capture_set
            or self.capture_report is not qualification.capture_report
            or self.device is not qualification.device
        ):
            raise ValueError("captured resident graph identities do not match.")


_ISSUED_CAPTURED_GRAPHS: dict[CapturedResidentGraph, object] = {}


def _require_issued_captured_graph(captured: object) -> CapturedResidentGraph:
    """Require a P2-issued record whose opaque handle retains identity."""
    if type(captured) is not CapturedResidentGraph:
        raise TypeError("captured must be an exact CapturedResidentGraph.")
    expected_handle = _ISSUED_CAPTURED_GRAPHS.get(captured)
    if expected_handle is None or expected_handle is not captured.handle:
        raise ValueError("captured resident graph handle is not P2-issued.")
    return captured


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
    """Qualify one exact prepared READY binding for later native capture.

    This metadata-only P1 boundary validates existing exact identities, then
    lazily resolves the adapter vocabulary in runtime, device, API, and
    callable order. It returns a fresh frozen record by reference only. It does
    not open a guard token; invoke native callables; capture, enqueue, allocate,
    synchronize, or transfer; create or release an opaque handle; register
    cleanup; or transition the READY lifecycle. Every rejection is read-only;
    P2/P3 own capture, handle lifetime, release, and cleanup.

    Args:
        binding: Exact attached READY resident graph-capture binding.
        prepared: Exact E8-F2 prepared resident simulation.
        capture_set: Exact E8-F3 published capture resource set.
        adapter: Caller-owned lazy native runtime adapter.

    Returns:
        A fresh qualification retaining only exact existing metadata identities
        and the adapter's exact callable vocabulary.

    Raises:
        TypeError: If a carrier, adapter member, probe result, or callable
            record has an invalid exact type.
        ValueError: If READY binding metadata, identity compatibility, or
            runtime/device/API qualification fails.
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
    prepared_any = cast("PreparedResidentSimulation", prepared)
    capture_set_any = cast("CaptureResourceSet", capture_set)
    request = cast("ResidentSimulationRequest", binding._request)
    session = cast("ResidentSession", binding._session)
    registry = cast("GPUResourceRegistry", binding._registry)
    guard = cast("ResidentStepGuard", binding._guard)
    lifecycle = binding._lifecycle
    signature = lifecycle.signature
    timestep = cast("PreparedResidentTimestep", prepared_any.timestep)
    if type(timestep) is not _prepared_resident_timestep_type():
        raise TypeError("timestep must be an exact PreparedResidentTimestep.")
    requirements = prepared_any.capture_requirements
    if type(requirements) is not _capture_resource_requirements_type():
        raise TypeError(
            "capture_requirements must be an exact CaptureResourceRequirements."
        )
    duration_is_identity = prepared_any.duration is timestep.duration
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
    if not duration_is_identity and prepared_any.duration != timestep.duration:
        raise ValueError("prepared graph-capture durations do not match.")
    from particula.execution.resident_scheduler import (
        _validate_prepared_resident_simulation,
    )

    # Qualification accepts only a scheduler-authoritative prepared record.
    # This shared validation covers every E8-F2 operation/product link before
    # any adapter member is read.
    _validate_prepared_resident_simulation(prepared)
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
    # Adapter code is caller-owned and may reenter resident metadata. Recheck
    # the complete prepared contract and the qualification snapshots before
    # retaining a callable vocabulary from a stale READY binding.
    _validate_prepared_resident_simulation(prepared)
    binding = _require_attached_resident_binding(binding)
    if (
        binding._request is not request
        or binding._session is not session
        or binding._registry is not registry
        or binding._guard is not guard
        or binding._lifecycle is not lifecycle
        or lifecycle.signature is not signature
        or binding._lifecycle.state is not GraphCaptureLifecycleState.READY
        or lifecycle.capability is not capability
    ):
        raise ValueError(
            "graph-capture qualification state changed during adapter callback."
        )
    guard.assert_step_closed()
    registry.validate_pinned_session(session)
    if session.lifecycle.name != "ACTIVE":
        raise ValueError("resident session must be ACTIVE.")
    if (
        capability.device != session.metadata.device
        or capability.availability is not GraphCaptureAvailability.AVAILABLE
        or capability.device.backend is not Backend.WARP
        or capability.device.native == "cpu"
    ):
        raise ValueError(
            "graph-capture qualification state changed during adapter callback."
        )
    compatibility = compare_resident_graph_capture_signature(
        signature,
        request,
        admission_token=signature,
    )
    if not compatibility.compatible:
        raise ValueError("resident graph-capture signature is incompatible.")
    if validate_resident_capture_resources(request) is not capture_set:
        raise ValueError("prepared capture resource set does not match.")
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


def _validate_prepared_graph_capture_qualification(
    qualification: object,
) -> PreparedGraphCaptureQualification:
    """Recheck one READY qualification without invoking native callables.

    Args:
        qualification: Candidate exact qualification retaining the prepared
            binding and native callable vocabulary.

    Returns:
        The same validated qualification object.

    Raises:
        TypeError: If ``qualification`` or a retained concrete carrier has an
            invalid exact type.
        ValueError: If the READY binding, resident session, signature, prepared
            operations, or published capture resource set has changed.
    """
    if type(qualification) is not PreparedGraphCaptureQualification:
        raise TypeError(
            "qualification must be an exact PreparedGraphCaptureQualification."
        )
    typed = cast("PreparedGraphCaptureQualification", qualification)
    binding = _require_attached_resident_binding(typed.binding)
    if (
        typed.lifecycle is not binding.lifecycle
        or typed.signature is not typed.lifecycle.signature
        or typed.request is not binding._request
        or typed.session is not binding._session
        or typed.registry is not binding._registry
        or typed.guard is not binding._guard
        or type(typed.prepared) is not _prepared_resident_simulation_type()
    ):
        raise ValueError("prepared graph-capture qualification changed.")
    prepared = cast("PreparedResidentSimulation", typed.prepared)
    session = cast("ResidentSession", typed.session)
    request = cast("ResidentSimulationRequest", typed.request)
    schedule = cast("ResolvedTimestepSchedule", request.schedule)
    if type(typed.timestep) is not _prepared_resident_timestep_type():
        raise TypeError("timestep must be an exact PreparedResidentTimestep.")
    timestep = cast("PreparedResidentTimestep", typed.timestep)
    if (
        typed.timestep is not prepared.timestep
        or typed.capture_requirements is not prepared.capture_requirements
        or typed.capture_set is not prepared.capture_set
        or typed.capture_report is not prepared.capture_report
        or typed.device is not session.metadata.device
        or typed.dimensions is not session.dimensions
        or typed.graph is not request.graph
        or typed.schedule is not request.schedule
        or typed.ordered_node_ids is not schedule.ordered_node_ids
        or typed.duration is not prepared.duration
        or typed.primary_arrays is not typed.signature.primary_arrays
        or typed.resource_views is not typed.signature.resource_views
        or type(typed.native_callables) is not GraphCaptureNativeCallables
        or (
            not typed.duration_is_identity
            and typed.duration != timestep.duration
        )
    ):
        raise ValueError("prepared graph-capture qualification changed.")
    from particula.execution.resident_scheduler import (
        _validate_prepared_resident_simulation,
    )

    _validate_prepared_resident_simulation(prepared)
    if typed.lifecycle.state is not GraphCaptureLifecycleState.READY:
        raise ValueError("graph capture lifecycle must be ready.")
    capability = typed.lifecycle.capability
    if (
        capability.device != session.metadata.device
        or capability.availability is not GraphCaptureAvailability.AVAILABLE
        or capability.device.backend is not Backend.WARP
        or capability.device.native == "cpu"
    ):
        raise ValueError("graph capture requires a CUDA resident device.")
    if (
        validate_resident_capture_resources(typed.request)
        is not typed.capture_set
    ):
        raise ValueError("prepared capture resource set does not match.")
    return typed


def _classify_capture_operational_failure(
    qualification: PreparedGraphCaptureQualification,
) -> BaseException | None:
    """Classify a capture failure when its exact binding remains attached.

    Args:
        qualification: Exact qualification associated with the failed capture.

    Returns:
        The classification error when classification fails, otherwise ``None``.
    """
    binding = qualification.binding
    if (
        type(binding) is not ResidentGraphCaptureBinding
        or binding._request is not qualification.request
        or getattr(qualification.request, "graph_capture_binding", None)
        is not binding
    ):
        return None
    try:
        classify_resident_graph_capture_writer_failure(binding)
    except BaseException as error:
        try:
            _fault_resident_graph_capture_after_classification_failure(binding)
        except BaseException:
            return error
        return error
    return None


def _raise_capture_operational_failure(
    qualification: PreparedGraphCaptureQualification,
    error: BaseException,
    cleanup_error: BaseException | None = None,
) -> NoReturn:
    """Raise an operational capture error while retaining its primary cause.

    Args:
        qualification: Exact qualification associated with the failed capture.
        error: Primary operational error to propagate.
        cleanup_error: Optional capture-end or release error to chain.

    Raises:
        BaseException: Always raises ``error`` with any cleanup or
            classification failure chained as its cause.
    """
    classification_error = _classify_capture_operational_failure(qualification)
    if cleanup_error is not None:
        if classification_error is not None:
            cleanup_error.__context__ = classification_error
        raise error from cleanup_error
    if classification_error is not None:
        raise error from classification_error
    raise error


def capture_prepared_resident_graph(  # noqa: C901
    qualification: object,
) -> CapturedResidentGraph:
    """Capture one qualified prepared timestep and retain its opaque handle.

    Only native ``capture_begin()``, retained prepared dispatch, and
    ``capture_end()`` run inside the capture window. A successful end handle is
    released exactly once only if post-end validation or lifecycle completion
    rejects it. This concrete-only operation neither instantiates, launches, nor
    replays a graph, and it performs no token, allocation, transfer, readback,
    or synchronization work in the capture window.

    Args:
        qualification: Exact READY qualification returned by
            :func:`qualify_prepared_resident_graph_capture`.

    Returns:
        Immutable record retaining the captured lifecycle successor and exact
        opaque handle returned by native ``capture_end()``.

    Raises:
        TypeError: If ``qualification`` is not an exact valid qualification.
        ValueError: If retained READY metadata has drifted, capture is already
            active for the binding, or lifecycle completion cannot publish a
            CAPTURED successor.
        BaseException: Propagates native begin, dispatch, end, or release errors
            while preserving the operational error as the primary cause.
    """
    try:
        typed = _validate_prepared_graph_capture_qualification(qualification)
        binding_id = id(typed.binding)
        with _ACTIVE_CAPTURE_LOCK:
            if binding_id in _ACTIVE_CAPTURE_BINDING_IDS:
                raise ValueError("graph capture is already active for binding.")
            _ACTIVE_CAPTURE_BINDING_IDS.add(binding_id)
        try:
            native = typed.native_callables
            native.capture_begin()
            try:
                from particula.execution.resident_scheduler import (
                    _enqueue_captured_prepared_operations,
                )

                _enqueue_captured_prepared_operations(typed.prepared)
            except BaseException as error:
                try:
                    cleanup_handle = native.capture_end()
                except BaseException as cleanup_error:
                    _raise_capture_operational_failure(
                        typed, error, cleanup_error
                    )
                try:
                    native.capture_release(cleanup_handle)
                except BaseException as cleanup_error:
                    _raise_capture_operational_failure(
                        typed, error, cleanup_error
                    )
                _raise_capture_operational_failure(typed, error)
            try:
                handle = native.capture_end()
            except BaseException as error:
                _raise_capture_operational_failure(typed, error)

            if handle is None:
                _raise_capture_operational_failure(
                    typed,
                    ValueError("native graph capture did not return a handle."),
                )

            try:
                typed = _validate_prepared_graph_capture_qualification(typed)
                lifecycle = complete_resident_graph_capture(typed.binding)
                if (
                    lifecycle is not typed.binding.lifecycle
                    or lifecycle.state
                    is not GraphCaptureLifecycleState.CAPTURED
                ):
                    raise ValueError(
                        "graph capture did not transition to captured."
                    )
                captured = CapturedResidentGraph(
                    qualification=typed,
                    binding=typed.binding,
                    lifecycle=lifecycle,
                    signature=typed.signature,
                    request=typed.request,
                    session=typed.session,
                    registry=typed.registry,
                    guard=typed.guard,
                    prepared=typed.prepared,
                    timestep=typed.timestep,
                    capture_requirements=typed.capture_requirements,
                    capture_set=typed.capture_set,
                    capture_report=typed.capture_report,
                    device=typed.device,
                    handle=handle,
                )
                _ISSUED_CAPTURED_GRAPHS[captured] = handle
                return captured
            except BaseException as error:
                try:
                    native.capture_release(handle)
                except BaseException as release_error:
                    _raise_capture_operational_failure(
                        typed, error, release_error
                    )
                _raise_capture_operational_failure(typed, error)
        finally:
            with _ACTIVE_CAPTURE_LOCK:
                _ACTIVE_CAPTURE_BINDING_IDS.discard(binding_id)
    except BaseException:
        raise


def _validate_replay_captured_graph(
    captured: object, duration: object
) -> PreparedGraphCaptureQualification:
    """Validate an issued record without inspecting resident payloads."""
    captured = _require_issued_captured_graph(captured)
    qualification = captured.qualification
    if type(qualification) is not PreparedGraphCaptureQualification:
        raise TypeError(
            "qualification must be an exact PreparedGraphCaptureQualification."
        )
    session = cast("ResidentSession", captured.session)
    request = cast(Any, captured.request)
    timestep = cast(Any, qualification.timestep)
    prepared = cast(Any, qualification.prepared)
    guard = cast("ResidentStepGuard", captured.guard)
    if (
        captured.binding is not qualification.binding
        or captured.signature is not qualification.signature
        or captured.request is not qualification.request
        or captured.session is not qualification.session
        or captured.registry is not qualification.registry
        or captured.guard is not qualification.guard
        or captured.prepared is not qualification.prepared
        or captured.timestep is not qualification.timestep
        or captured.capture_requirements
        is not qualification.capture_requirements
        or captured.capture_set is not qualification.capture_set
        or captured.capture_report is not qualification.capture_report
        or captured.device is not qualification.device
        or captured.lifecycle is not captured.binding.lifecycle
        or captured.lifecycle.state is not GraphCaptureLifecycleState.CAPTURED
        or qualification.signature is not captured.lifecycle.signature
        or qualification.binding is not captured.binding
        or qualification.request is not captured.request
        or qualification.session is not captured.session
        or qualification.registry is not captured.registry
        or qualification.guard is not captured.guard
        or qualification.prepared is not captured.prepared
        or qualification.timestep is not captured.timestep
        or qualification.capture_requirements
        is not captured.capture_requirements
        or qualification.capture_set is not captured.capture_set
        or qualification.capture_report is not captured.capture_report
        or qualification.device is not captured.device
        or qualification.dimensions is not session.dimensions
        or qualification.graph is not request.graph
        or qualification.schedule is not request.schedule
        or qualification.ordered_node_ids
        is not request.schedule.ordered_node_ids
        or qualification.primary_arrays is not captured.signature.primary_arrays
        or qualification.resource_views is not captured.signature.resource_views
        or type(qualification.native_callables)
        is not GraphCaptureNativeCallables
    ):
        raise ValueError("captured resident graph identities do not match.")
    gate_resident_graph_capture(captured.binding)
    guard._validate_duration(duration)
    if (
        qualification.device != session.metadata.device
        or qualification.device != captured.lifecycle.capability.device
        or qualification.timestep is not prepared.timestep
        or qualification.duration is not prepared.duration
    ):
        raise ValueError("captured resident graph qualification changed.")
    if qualification.duration_is_identity:
        if (
            qualification.duration is not timestep.duration
            or duration is not qualification.duration
        ):
            raise ValueError("captured graph replay duration does not match.")
    elif (
        qualification.duration != timestep.duration
        or duration != qualification.duration
    ):
        raise ValueError("captured graph replay duration does not match.")
    return qualification


def _raise_replay_operational_failure(
    qualification: PreparedGraphCaptureQualification,
    token: object,
    error: BaseException,
) -> NoReturn:
    """Clean up a post-launch replay failure while preserving ``error``."""
    from particula.execution.gpu_session import (
        ResidentStepToken,
        _handle_failed_resident_operation,
        _ResidentOperationOutcome,
    )

    cleanup_error: BaseException | None = None
    try:
        _handle_failed_resident_operation(
            cast("ResidentSession", qualification.session),
            cast("GPUResourceRegistry", qualification.registry),
            cast("ResidentStepGuard", qualification.guard),
            cast("ResidentStepToken", token),
            _ResidentOperationOutcome.WRITER_MAY_HAVE_LAUNCHED,
        )
    except BaseException as caught:
        cleanup_error = caught
    classification_error: BaseException | None = None
    try:
        classify_resident_graph_capture_writer_failure(qualification.binding)
    except BaseException as caught:
        classification_error = caught
        try:
            _fault_resident_graph_capture_after_classification_failure(
                qualification.binding
            )
        except BaseException:  # noqa: S110
            pass
    if cleanup_error is not None:
        if classification_error is not None:
            cleanup_error.__context__ = classification_error
        raise error from cleanup_error
    if classification_error is not None:
        raise error from classification_error
    raise error


def replay_captured_resident_graph(captured: object, duration: object) -> None:
    """Launch one authentic captured graph under exactly one resident token.

    Accepted replay performs no prepared-host dispatch, native capture lifecycle
    work, allocation, payload scan, transfer, readback, synchronization, RNG
    reset, fallback, retry, or recapture. Pinned array values and resident RNG
    words may change while their retained identities remain compatible.
    """
    qualification = _validate_replay_captured_graph(captured, duration)
    captured_graph = cast(CapturedResidentGraph, captured)
    guard = cast("ResidentStepGuard", qualification.guard)
    token = guard.begin_step(duration)
    try:
        qualification.native_callables.capture_launch(captured_graph.handle)
        guard.complete_step(token)
    except BaseException as error:
        _raise_replay_operational_failure(qualification, token, error)


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
