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


def _registry_type() -> type[object]:
    """Return the concrete registry type without an import cycle."""
    from particula.execution.gpu_resources import GPUResourceRegistry

    return GPUResourceRegistry


@dataclass(frozen=True, eq=False)
class ResidentCommunicationRequest:
    """Bind exact resident resources to the two closed barrier nodes."""

    session: ResidentSession
    registry: object
    graph: ResolvedProcessGraph
    resources: CommunicationResources
    communication_node: ProcessNode
    volume_evolution_node: ProcessNode
    duration: Real

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
    """Validate metadata and dispatch resident barrier primitives once."""

    def __init__(self, request: ResidentCommunicationRequest) -> None:
        """Retain one exact communication request."""
        if type(request) is not ResidentCommunicationRequest:
            raise TypeError(
                "request must be an exact ResidentCommunicationRequest."
            )
        self._request = request

    def validate(self) -> None:
        """Validate identity and metadata without P1 scans or allocation."""
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
        """Dispatch exactly one native communication primitive by mode."""
        self.validate()
        request = self._request
        from particula.gpu.kernels.communication import (
            gas_communication_step_gpu,
            particle_communication_step_gpu,
        )

        mode = request.resources.configuration.communication_map.transport_mode
        if mode is CommunicationTransportMode.GAS:
            return gas_communication_step_gpu(
                request.session.particles,
                request.session.gas,
                request.resources.configuration,
                request.duration,
                request.resources.buffers,
            )
        if mode is CommunicationTransportMode.PARTICLES:
            return particle_communication_step_gpu(
                request.session.particles,
                request.resources.configuration,
                request.duration,
                request.resources.buffers,
            )
        raise ValueError(
            "resident communication supports GAS or PARTICLES only."
        )

    def execute_volume_evolution(self) -> object | None:
        """Apply the optional prescribed-volume writer without replacement."""
        self.validate()
        final_volumes = self._request.resources.final_volumes
        if final_volumes is None:
            return None
        from particula.gpu.kernels.communication import (
            volume_evolution_step_gpu,
        )

        return volume_evolution_step_gpu(
            self._request.session.particles,
            self._request.session.gas,
            final_volumes,
        )
