"""Copy closed-protocol resident diagnostics into caller-owned Warp outputs.

This direct-import-only module has no callback registration or package export.
It snapshots only resident gas concentration and environment saturation ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import warp as wp

from particula.execution.gpu_session import ResidentSession
from particula.execution.process_graph import (
    NodeKind,
    ProcessNode,
    ResolvedProcessGraph,
    ResourceRequirement,
    _is_resolver_produced_graph,
    resolve_canonical_topological_order,
)
from particula.execution.scheduler import ResolvedTimestepSchedule


class ResidentDiagnosticOperation(str, Enum):
    """Enumerate the only resident diagnostic snapshot operations.

    The protocol permits snapshots of current gas concentration and saturation
    ratio only. It does not accept callbacks or arbitrary resident inspection.
    """

    GAS_CONCENTRATION_SNAPSHOT = "gas_concentration_snapshot"
    SATURATION_RATIO_SNAPSHOT = "saturation_ratio_snapshot"


@dataclass(frozen=True, eq=False)
class ResidentDiagnosticRegistration:
    """Bind one closed diagnostic operation to one caller-owned output.

    Attributes:
        operation: Exact closed operation that selects the resident source.
        output: Caller-owned Warp ``float64`` array validated by the executor.
    """

    operation: ResidentDiagnosticOperation
    output: object

    def __post_init__(self) -> None:
        """Validate the exact closed diagnostic operation.

        Raises:
            TypeError: If ``operation`` is not an exact supported operation.
        """
        if type(self.operation) is not ResidentDiagnosticOperation:
            raise TypeError(
                "operation must be an exact ResidentDiagnosticOperation."
            )


@dataclass(frozen=True, eq=False)
class ResidentDiagnosticsPlan:
    """Bind ordered closed diagnostics to one resident graph and schedule.

    Attributes:
        session: Exact active resident session that owns diagnostic sources.
        registry: Exact registry pinned to ``session``.
        graph: Resolver-produced graph containing ``node`` by identity.
        schedule: Matching resolved schedule that ends with ``node``.
        node: Canonical ``diagnostics`` process node.
        registrations: Ordered closed operation and output bindings.
    """

    session: ResidentSession
    registry: object
    graph: ResolvedProcessGraph
    schedule: ResolvedTimestepSchedule
    node: ProcessNode
    registrations: tuple[ResidentDiagnosticRegistration, ...]

    def __post_init__(self) -> None:
        """Validate exact types for the resident diagnostics binding.

        Structural graph, lifecycle, and output validation is deferred to the
        executor so plan construction does not inspect Warp-array metadata.

        Raises:
            TypeError: If a carrier or registration has an inexact type.
        """
        from particula.execution.gpu_resources import GPUResourceRegistry

        if type(self.session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        if type(self.registry) is not GPUResourceRegistry:
            raise TypeError("registry must be an exact GPUResourceRegistry.")
        if type(self.graph) is not ResolvedProcessGraph:
            raise TypeError("graph must be an exact ResolvedProcessGraph.")
        if type(self.schedule) is not ResolvedTimestepSchedule:
            raise TypeError(
                "schedule must be an exact ResolvedTimestepSchedule."
            )
        if type(self.node) is not ProcessNode:
            raise TypeError("node must be an exact ProcessNode.")
        if type(self.registrations) is not tuple or not all(
            type(item) is ResidentDiagnosticRegistration
            for item in self.registrations
        ):
            raise TypeError(
                "registrations must be exact "
                "ResidentDiagnosticRegistration tuple."
            )


@wp.kernel
def _copy_snapshot(source: Any, output: Any) -> None:
    """Copy one resident diagnostic matrix element to its output."""
    box, species = wp.tid()  # type: ignore[misc]
    output[box, species] = source[box, species]


class ResidentDiagnosticsExecutor:
    """Execute an already-bound closed diagnostics plan without transfers.

    Validation preserves caller ownership and rejects outputs that alias
    resident primaries, published sidecars, or another diagnostic output.
    Execution copies registrations in declared order and is write-free for
    empty matrices.
    """

    def _validate(self, plan: ResidentDiagnosticsPlan) -> None:
        """Validate plan provenance, closed operation order, and outputs.

        Args:
            plan: Exact diagnostics plan whose retained bindings are checked.

        Raises:
            ValueError: If lifecycle, graph, schedule, protocol, or output
                metadata validation fails.
        """
        registry = cast(Any, plan.registry)
        if registry._session is not plan.session:
            raise ValueError("diagnostics registry must be bound to session.")
        registry.validate_pinned_session(plan.session)
        if not _is_resolver_produced_graph(plan.graph):
            raise ValueError(
                "diagnostics graph must be produced by plan resolution."
            )
        if not any(node is plan.node for node in plan.graph.nodes):
            raise ValueError("diagnostics node must be a graph member.")
        if not any(node is plan.node for node in plan.schedule.nodes):
            raise ValueError("diagnostics node must be a schedule member.")
        if (
            plan.node.node_id != "diagnostics"
            or plan.node.kind is not NodeKind.DIAGNOSTIC
        ):
            raise ValueError("diagnostics node has an invalid canonical role.")
        if plan.node.resources != frozenset(
            {
                ResourceRequirement.PARTICLES,
                ResourceRequirement.GAS,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.THERMODYNAMICS,
                ResourceRequirement.DIAGNOSTICS,
            }
        ):
            raise ValueError("diagnostics node has an invalid canonical role.")
        if plan.schedule.ordered_node_ids[-1:] != ("diagnostics",):
            raise ValueError("diagnostics must be the final scheduled node.")
        if (
            plan.schedule.ordered_node_ids
            != resolve_canonical_topological_order(
                plan.schedule.nodes, plan.schedule.dependencies
            )
        ):
            raise ValueError("diagnostics schedule must be canonical.")
        operations = tuple(item.operation for item in plan.registrations)
        if operations != tuple(ResidentDiagnosticOperation):
            raise ValueError(
                "diagnostic operations must be unique and match the closed "
                "canonical tuple."
            )
        registry.validate_diagnostic_outputs(
            plan.session, tuple(item.output for item in plan.registrations)
        )

    def validate(self, plan: object) -> ResidentDiagnosticsPlan:
        """Validate one exact diagnostics plan without dispatching a kernel.

        Returns:
            The unchanged, exact validated plan.
        """
        if type(plan) is not ResidentDiagnosticsPlan:
            raise TypeError("plan must be an exact ResidentDiagnosticsPlan.")
        self._validate(plan)
        return plan

    def execute(self, plan: object) -> None:
        """Validate and snapshot each registration in declared order.

        Empty ``(B, S)`` output schemas complete without a kernel launch.

        Args:
            plan: Exact plan selecting the sources and caller-owned outputs.

        Raises:
            TypeError: If ``plan`` is not an exact diagnostics plan.
            ValueError: If its bindings or output metadata are invalid.
        """
        plan = self.validate(plan)
        dimensions = plan.session.dimensions
        if not dimensions.n_boxes or not dimensions.n_species:
            return
        gas = cast(Any, plan.session.gas)
        environment = cast(Any, plan.session.environment)
        for registration in plan.registrations:
            source = (
                gas.concentration
                if registration.operation
                is ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT
                else environment.saturation_ratio
            )
            wp.launch(
                _copy_snapshot,
                dim=(dimensions.n_boxes, dimensions.n_species),
                inputs=[source, registration.output],
                device=source.device,
            )
