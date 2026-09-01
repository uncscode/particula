"""Compose pinned communication resources into resident barrier calls.

This concrete-only module dispatches already acquired closed-map communication
and optional volume evolution. Its shared validator checks retained metadata
without constructing an executor or mutating resident state. It never performs
P1 validation, acquisition, host conversion, synchronization, fallback, or
recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

from particula.execution import _isfinite_real
from particula.execution.communication import CommunicationTransportMode
from particula.execution.gpu_resources import CommunicationResources
from particula.execution.gpu_session import ResidentSession
from particula.execution.process_graph import (
    NodeKind,
    ProcessNode,
    ResolvedProcessGraph,
    _is_resolver_produced_graph,
)
from particula.execution.resident_enqueue import (
    PreparedResidentTimestep,
    _validate_ready_attachment,
)
from particula.gpu.kernels.communication import (
    GasCommunicationBuffers,
    ParticleCommunicationBuffers,
    _enqueue_prepared_resident_gas_communication,
    _enqueue_prepared_resident_particle_communication,
    _enqueue_prepared_resident_volume_evolution,
)


def _registry_type() -> type[object]:
    """Return the concrete registry type without an import cycle.

    Returns:
        The direct-module-only GPU resource registry type.
    """
    from particula.execution.gpu_resources import GPUResourceRegistry

    return GPUResourceRegistry


@dataclass(frozen=True, eq=False)
class ResidentCommunicationRequest:
    """Bind exact resident resources to the two closed barrier nodes.

    The request retains the session, registry, graph, published communication
    view, and graph-node objects by identity. It carries a finite nonnegative
    duration but does not validate payload physics, allocate, transfer,
    synchronize, or mutate resident state at construction.

    Attributes:
        session: Exact active resident session whose containers are dispatched.
        registry: Exact registry that published ``resources``.
        graph: Resolver-produced graph owning the two barrier nodes.
        resources: Exact published closed-map communication resource view.
        communication_node: Exact ``communication`` graph node.
        volume_evolution_node: Exact ``volume_evolution`` graph node.
        duration: Finite nonnegative barrier duration in s.
    """

    session: ResidentSession
    registry: object
    graph: ResolvedProcessGraph
    resources: CommunicationResources
    communication_node: ProcessNode
    volume_evolution_node: ProcessNode
    duration: float

    def __post_init__(self) -> None:
        """Validate exact carrier types only."""
        exact = (
            (self.session, ResidentSession, "session"),
            (self.registry, _registry_type(), "registry"),
            (self.graph, ResolvedProcessGraph, "graph"),
            (self.resources, CommunicationResources, "resources"),
            (self.communication_node, ProcessNode, "communication_node"),
            (self.volume_evolution_node, ProcessNode, "volume_evolution_node"),
        )
        for value, expected, name in exact:
            if type(value) is not expected:
                raise TypeError(f"{name} must be an exact {expected.__name__}.")


@dataclass(frozen=True, eq=False)
class PreparedResidentCommunicationBinding:
    """Freeze a validated closed communication barrier for device enqueue.

    This concrete-only carrier retains P1, request, primary, map, work, and
    optional-volume identities after setup-time validation. Enqueue performs no
    lookup, validation, allocation, transfer, readback, synchronization, or
    fallback. Only closed GAS or PARTICLES maps are supported. Equal final
    volumes are a write-free barrier; after a changed-volume device writer
    launches, rollback is not promised.

    Attributes:
        prepared_timestep: Exact READY P1 timestep retained by identity.
        request: Validated resident communication request retained by identity.
        particles: Resident Warp particle container.
        gas: Resident Warp gas container.
        masses: Resident particle mass array.
        particle_concentration: Resident particle concentration array.
        density: Resident particle density array.
        volume: Resident per-box volume array.
        charge: Resident particle charge array.
        gas_concentration: Resident gas concentration array.
        source_boxes: Closed-map source endpoint array.
        destination_boxes: Closed-map destination endpoint array.
        enabled: Communication edge enablement array.
        rates: Communication edge-rate array.
        configuration: Validated communication configuration.
        map_form: Communication map dimensionality declaration.
        one_dimensional: Integer flag for one-dimensional planning.
        mode: GAS or PARTICLES transport mode.
        edge_capacity: Number of retained map edge slots.
        duration: Nonnegative resident barrier duration in seconds.
        device: Device hosting all retained Warp arrays.
        dimensions: Fixed resident particle and gas dimensions.
        buffers: Mode-specific caller-owned work buffers.
        invalid: Registry-pinned validation status array.
        active_or_demand: Registry-pinned activity or demand status array.
        volume_invalid: Registry-pinned volume validation status array.
        volume_changed: Registry-pinned volume-change status array.
        initial_masses: Particle commit snapshot, when retained.
        initial_concentration: Particle concentration snapshot, when retained.
        initial_charge: Particle charge snapshot, when retained.
        final_volumes: Optional prescribed final-volume array.
    """

    prepared_timestep: PreparedResidentTimestep
    request: ResidentCommunicationRequest
    particles: object
    gas: object
    masses: object
    particle_concentration: object
    density: object
    volume: object
    charge: object
    gas_concentration: object
    source_boxes: object
    destination_boxes: object
    enabled: object
    rates: object
    configuration: object
    map_form: object
    one_dimensional: int
    mode: CommunicationTransportMode
    edge_capacity: int
    duration: float
    device: object
    dimensions: object
    buffers: GasCommunicationBuffers | ParticleCommunicationBuffers
    invalid: object
    active_or_demand: object
    volume_invalid: object
    volume_changed: object
    initial_masses: object | None
    initial_concentration: object | None
    initial_charge: object | None
    final_volumes: object | None


def setup_prepared_resident_communication(  # noqa: C901
    prepared_timestep: object, request: object
) -> PreparedResidentCommunicationBinding:
    """Validate P1/request identities and freeze a closed native barrier.

    Setup is read-only and may perform registry/request metadata validation. The
    returned concrete-only binding retains all dispatch inputs by identity for a
    later enqueue; it neither opens a guard nor launches a native writer.

    Args:
        prepared_timestep: Exact READY P1 timestep for the complete request.
        request: Exact resident communication request attached to that P1 input.

    Returns:
        Frozen identity-semantic prepared communication binding.

    Raises:
        TypeError: If either supplied carrier has an invalid exact type.
        ValueError: If P1, resource, graph, primary, or map bindings drift.
    """
    if type(prepared_timestep) is not PreparedResidentTimestep:
        raise TypeError(
            "prepared_timestep must be an exact PreparedResidentTimestep."
        )
    if type(request) is not ResidentCommunicationRequest:
        raise TypeError(
            "request must be an exact ResidentCommunicationRequest."
        )
    prepared = cast(PreparedResidentTimestep, prepared_timestep)
    typed = cast(ResidentCommunicationRequest, request)
    _validate_ready_attachment(
        prepared.request,
        prepared.binding,
        prepared.lifecycle,
        prepared.signature,
        prepared.session,
        prepared.registry,
        prepared.guard,
    )
    if prepared.request.communication is not typed:
        raise ValueError(
            "prepared timestep does not retain the supplied request."
        )
    if (
        prepared.session is not typed.session
        or prepared.registry is not typed.registry
        or prepared.graph is not typed.graph
        or prepared.dimensions is not typed.session.dimensions
        or prepared.duration != typed.duration
    ):
        raise ValueError("prepared timestep identities do not match request.")
    validate_resident_communication_request(typed)
    if (
        typed.communication_node.node_id != "communication"
        or typed.volume_evolution_node.node_id != "volume_evolution"
        or "communication" not in prepared.ordered_node_ids
        or "volume_evolution" not in prepared.ordered_node_ids
        or prepared.ordered_node_ids.index("communication")
        >= prepared.ordered_node_ids.index("volume_evolution")
    ):
        raise ValueError("prepared communication nodes do not match schedule.")
    particles = cast(Any, typed.session.particles)
    gas = cast(Any, typed.session.gas)
    primaries = (
        particles.masses,
        particles.concentration,
        particles.density,
        particles.volume,
        particles.charge,
        gas.molar_mass,
        gas.concentration,
        gas.partitioning,
        gas.vapor_pressure,
        cast(Any, typed.session.environment).temperature,
        cast(Any, typed.session.environment).pressure,
        cast(Any, typed.session.environment).saturation_ratio,
    )
    if len(prepared.primary_arrays) != len(primaries) or any(
        left is not right
        for left, right in zip(prepared.primary_arrays, primaries, strict=True)
    ):
        raise ValueError(
            "prepared timestep primary arrays do not match session."
        )
    if not any(view is typed.resources for view in prepared.resource_views):
        raise ValueError(
            "prepared timestep does not retain communication resources."
        )
    configuration = typed.resources.configuration
    map_data = configuration.communication_map
    mode = map_data.transport_mode
    if mode not in (
        CommunicationTransportMode.GAS,
        CommunicationTransportMode.PARTICLES,
    ):
        raise ValueError(
            "resident communication supports GAS or PARTICLES only."
        )
    if map_data.form.name not in {"ONE_DIMENSIONAL", "TWO_DIMENSIONAL"}:
        raise ValueError("resident communication map form is unsupported.")
    resources = typed.resources
    state = resources.execution_state
    if mode is CommunicationTransportMode.GAS:
        if type(resources.buffers) is not GasCommunicationBuffers:
            raise ValueError("communication buffers must match transport mode.")
    else:
        if (
            type(resources.buffers) is not ParticleCommunicationBuffers
            or state.initial_masses is None
            or state.initial_concentration is None
            or state.initial_charge is None
        ):
            raise ValueError(
                "particle communication snapshots must be complete."
            )
    return PreparedResidentCommunicationBinding(
        prepared,
        typed,
        particles,
        gas,
        particles.masses,
        particles.concentration,
        particles.density,
        particles.volume,
        particles.charge,
        gas.concentration,
        map_data.source_boxes,
        map_data.destination_boxes,
        map_data.enabled,
        map_data.rates,
        configuration,
        map_data.form,
        int(map_data.form.name == "ONE_DIMENSIONAL"),
        mode,
        int(map_data.edge_capacity),
        typed.duration,
        particles.masses.device,
        typed.session.dimensions,
        resources.buffers,
        state.invalid,
        state.active_or_demand,
        state.volume_invalid,
        state.volume_changed,
        state.initial_masses,
        state.initial_concentration,
        state.initial_charge,
        resources.final_volumes,
    )


def _enqueue_prepared_resident_communication(
    binding: PreparedResidentCommunicationBinding,
) -> object | None:
    """Enqueue the retained communication barrier and optional volume barrier.

    The binding is assumed to have passed setup validation. This function only
    dispatches the already-bound native helpers in communication-first order;
    it performs no lookup, allocation, validation, transfer, synchronization,
    or host inspection. A device writer has no rollback guarantee.

    Args:
        binding: Frozen resident communication inputs and caller-owned buffers.

    Returns:
        The mode-specific communication result, or ``None`` only when the
        selected native helper reports no result.
    """
    if binding.mode is CommunicationTransportMode.GAS:
        result: object = _enqueue_prepared_resident_gas_communication(
            binding.particles,
            binding.gas,
            binding.source_boxes,
            binding.destination_boxes,
            binding.enabled,
            binding.rates,
            binding.edge_capacity,
            binding.duration,
            binding.device,
            cast(GasCommunicationBuffers, binding.buffers),
            binding.invalid,
            binding.active_or_demand,
        )
    else:
        result = _enqueue_prepared_resident_particle_communication(
            binding.particles,
            binding.source_boxes,
            binding.destination_boxes,
            binding.enabled,
            binding.rates,
            binding.edge_capacity,
            binding.one_dimensional,
            binding.duration,
            binding.device,
            cast(ParticleCommunicationBuffers, binding.buffers),
            binding.invalid,
            binding.active_or_demand,
            binding.initial_masses,
            binding.initial_concentration,
            binding.initial_charge,
        )
    if binding.final_volumes is not None:
        _enqueue_prepared_resident_volume_evolution(
            binding.particles,
            binding.gas,
            binding.final_volumes,
            binding.volume_invalid,
            binding.volume_changed,
            binding.device,
        )
    return result


class ResidentCommunicationExecutor:
    """Validate metadata and dispatch resident barrier primitives once.

    The executor preserves the closed order: communication uses pre-update
    volumes, then optional volume evolution applies prescribed final volumes.
    Neither dispatch path transfers, synchronizes, retries, or recovers from a
    native writer failure.
    """

    def __init__(self, request: ResidentCommunicationRequest) -> None:
        """Retain one exact communication request.

        Args:
            request: Identity-bound closed-map resident barrier request.

        Raises:
            TypeError: If ``request`` is not an exact
                ``ResidentCommunicationRequest``.
        """
        if type(request) is not ResidentCommunicationRequest:
            raise TypeError(
                "request must be an exact ResidentCommunicationRequest."
            )
        self._request = request

    def validate(self) -> None:
        """Validate identity and metadata without P1 scans or allocation.

        Validation delegates only to the registry's metadata seam and verifies
        resolver provenance plus exact barrier-node identities. It performs no
        configuration acquisition or P1 scan, payload readback, transfer,
        synchronization, allocation, primitive dispatch, or mutation.

        Raises:
            TypeError: If duration is not a non-boolean real value.
            ValueError: If duration, registry binding, graph provenance, or
                barrier-node identity and kind are invalid.
        """
        validate_resident_communication_request(self._request)

    def execute_communication(self) -> object:
        """Dispatch exactly one native communication primitive by mode.

        The selected GAS or PARTICLES primitive receives resident containers,
        configuration, duration, and work record by identity. This method never
        replaces those objects or transfers, reads back, synchronizes, retries,
        falls back, acquires resources, or rolls back after a native writer
        launches. Prelaunch validation errors occur before primitive dispatch;
        native errors propagate unchanged.

        Returns:
            The selected native primitive's return value.

        Raises:
            TypeError: If request validation finds an invalid duration type.
            ValueError: If metadata validation fails or the mode is unsupported.
            Exception: Propagates a native primitive error without recovery.
        """
        self.validate()
        request = self._request
        from particula.gpu.kernels.communication import (
            resident_gas_communication_step_gpu,
            resident_particle_communication_step_gpu,
        )

        mode = request.resources.configuration.communication_map.transport_mode
        if mode is CommunicationTransportMode.GAS:
            return resident_gas_communication_step_gpu(
                request.session.particles,
                request.session.gas,
                request.resources.configuration,
                request.duration,
                cast(GasCommunicationBuffers, request.resources.buffers),
                request.resources.execution_state.invalid,
                request.resources.execution_state.active_or_demand,
            )
        if mode is CommunicationTransportMode.PARTICLES:
            state = request.resources.execution_state
            return resident_particle_communication_step_gpu(
                request.session.particles,
                request.resources.configuration,
                request.duration,
                cast(ParticleCommunicationBuffers, request.resources.buffers),
                state.invalid,
                state.active_or_demand,
                state.initial_masses,
                state.initial_concentration,
                state.initial_charge,
            )
        raise ValueError(
            "resident communication supports GAS or PARTICLES only."
        )

    def execute_volume_evolution(self) -> object | None:
        """Apply the optional prescribed-volume writer without replacement.

        If present, final volumes and resident particle and gas containers are
        passed to the native writer by identity. If absent, this is a successful
        write-free return. The boundary performs no object replacement,
        transfer, readback, synchronization, acquisition, retry, fallback, or
        rollback after launch.

        Returns:
            ``None`` when no final volumes are pinned; otherwise the native
            volume primitive's return value.

        Raises:
            TypeError: If request validation finds an invalid duration type.
            ValueError: If the resident communication binding is invalid.
            Exception: Propagates a native volume-writer error without recovery.
        """
        self.validate()
        final_volumes = self._request.resources.final_volumes
        if final_volumes is None:
            return None
        from particula.gpu.kernels.communication import (
            resident_volume_evolution_step_gpu,
        )

        return resident_volume_evolution_step_gpu(
            self._request.session.particles,
            self._request.session.gas,
            final_volumes,
            self._request.resources.execution_state.volume_invalid,
            self._request.resources.execution_state.volume_changed,
        )


def validate_resident_communication_request(
    request: object,
) -> ResidentCommunicationRequest:
    """Validate one resident communication request without an executor.

    This read-only shared validation seam performs no resource acquisition,
    payload inspection, primitive dispatch, transfer, synchronization, or
    resident-state mutation.

    Args:
        request: Candidate exact resident communication request.

    Returns:
        The unchanged validated request.

    Raises:
        TypeError: If the request or its duration has an invalid type.
        ValueError: If metadata, duration, or barrier nodes are invalid.
    """
    if type(request) is not ResidentCommunicationRequest:
        raise TypeError(
            "request must be an exact ResidentCommunicationRequest."
        )
    if isinstance(request.duration, bool) or not isinstance(
        request.duration, Real
    ):
        raise TypeError("duration must be a non-boolean real.")
    if not _isfinite_real(request.duration) or request.duration < 0:
        raise ValueError("duration must be finite and nonnegative.")
    registry = cast(Any, request.registry)
    registry.validate_communication_resources(
        request.session, request.resources
    )
    if not _is_resolver_produced_graph(request.graph):
        raise ValueError("graph must be produced by plan resolution.")
    nodes = {node.node_id: node for node in request.graph.nodes}
    if (
        nodes.get("communication") is not request.communication_node
        or nodes.get("volume_evolution") is not request.volume_evolution_node
        or request.communication_node.kind is not NodeKind.COMMUNICATION
        or request.volume_evolution_node.kind is not NodeKind.VOLUME_EVOLUTION
    ):
        raise ValueError("communication barrier nodes do not match graph.")
    return request
