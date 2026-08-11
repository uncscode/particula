"""Compose pinned communication resources into resident barrier calls.

This concrete-only module dispatches already acquired closed-map communication
and optional volume evolution.  It never performs P1 validation, acquisition,
host conversion, synchronization, fallback, or recovery.
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
from particula.gpu.kernels.communication import (
    GasCommunicationBuffers,
    ParticleCommunicationBuffers,
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
        request = self._request
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
            or nodes.get("volume_evolution")
            is not request.volume_evolution_node
            or request.communication_node.kind is not NodeKind.COMMUNICATION
            or request.volume_evolution_node.kind
            is not NodeKind.VOLUME_EVOLUTION
        ):
            raise ValueError("communication barrier nodes do not match graph.")

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
        if request.resources.all_disabled:
            return request.session.particles, request.session.gas
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
