"""Write closed-protocol resident diagnostics into caller-owned Warp outputs.

This direct-import-only module has no callback registration or package export.
Its canonical order is gas and saturation snapshots, total species mass,
particle-number concentration, latent heat energy, and conservation residual.
Matrix operations use ``(B, S)`` float64 arrays; particle number uses ``(B,)``.
Masses are extensive kg, energy is signed J, and residual is
``total - baseline - source + sink``.  Execution has no host readback,
synchronization, transfer, allocation, or physics mutation.
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
from particula.execution.scheduler import (
    ResolvedTimestepSchedule,
    is_resolver_produced_schedule,
)


class ResidentDiagnosticOperation(str, Enum):
    """Enumerate the closed resident diagnostic operations in launch order."""

    GAS_CONCENTRATION_SNAPSHOT = "gas_concentration_snapshot"
    SATURATION_RATIO_SNAPSHOT = "saturation_ratio_snapshot"
    TOTAL_SPECIES_MASS = "total_species_mass"
    PARTICLE_NUMBER_CONCENTRATION = "particle_number_concentration"
    LATENT_HEAT_ENERGY = "latent_heat_energy"
    CONSERVATION_RESIDUAL = "conservation_residual"


@dataclass(frozen=True, eq=False)
class ResidentDiagnosticRegistration:
    """Bind one closed diagnostic operation to caller-owned Warp arrays.

    Attributes:
        operation: Exact closed operation that selects the resident source.
        output: Caller-owned Warp ``float64`` array validated by the executor.
        energy_transfer: Required signed energy input for latent-energy output.
        baseline_total_mass: Required extensive baseline for residual output.
        source_ledger: Required nonnegative extensive source ledger for the
            residual.
        sink_ledger: Required nonnegative extensive sink ledger for residual.
    """

    operation: ResidentDiagnosticOperation
    output: object
    energy_transfer: object | None = None
    baseline_total_mass: object | None = None
    source_ledger: object | None = None
    sink_ledger: object | None = None

    def __post_init__(self) -> None:
        """Validate the exact closed diagnostic operation.

        Raises:
            TypeError: If ``operation`` is not an exact supported operation.
        """
        if type(self.operation) is not ResidentDiagnosticOperation:
            raise TypeError(
                "operation must be an exact ResidentDiagnosticOperation."
            )
        inputs = (
            self.energy_transfer,
            self.baseline_total_mass,
            self.source_ledger,
            self.sink_ledger,
        )
        if self.operation is ResidentDiagnosticOperation.LATENT_HEAT_ENERGY:
            if self.energy_transfer is None or any(
                item is not None for item in inputs[1:]
            ):
                raise ValueError("latent energy requires only energy_transfer.")
        elif (
            self.operation is ResidentDiagnosticOperation.CONSERVATION_RESIDUAL
        ):
            if (
                any(item is None for item in inputs[1:])
                or self.energy_transfer is not None
            ):
                raise ValueError(
                    "residual requires baseline, source, and sink ledgers."
                )
        elif any(item is not None for item in inputs):
            raise ValueError("diagnostic operation forbids accounting inputs.")


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


@wp.kernel
def _total_species_mass(
    masses: Any,
    concentration: Any,
    gas_concentration: Any,
    volume: Any,
    output: Any,
) -> None:
    """Reduce concentration-weighted particle and gas mass for one lane."""
    box, species = wp.tid()  # type: ignore[misc]
    total = gas_concentration[box, species]
    for particle in range(masses.shape[1]):
        total += masses[box, particle, species] * concentration[box, particle]
    output[box, species] = volume[box] * total


@wp.kernel
def _particle_number(concentration: Any, output: Any) -> None:
    """Reduce particle number concentration for one box."""
    box = wp.tid()  # type: ignore[misc]
    total = wp.float64(0.0)
    for particle in range(concentration.shape[1]):
        total += concentration[box, particle]
    output[box] = total


@wp.kernel
def _conservation_residual(
    masses: Any,
    concentration: Any,
    gas_concentration: Any,
    volume: Any,
    baseline: Any,
    source: Any,
    sink: Any,
    output: Any,
) -> None:
    """Write the ledger-aware extensive mass residual for one lane."""
    box, species = wp.tid()  # type: ignore[misc]
    total = gas_concentration[box, species]
    for particle in range(masses.shape[1]):
        total += masses[box, particle, species] * concentration[box, particle]
    output[box, species] = (
        volume[box] * total
        - baseline[box, species]
        - source[box, species]
        + sink[box, species]
    )


class ResidentDiagnosticsExecutor:
    """Execute an already-bound closed diagnostics plan without transfers.

    Validation preserves caller ownership and rejects outputs that alias
    resident primaries, published sidecars, or another diagnostic output.
    Execution copies registrations in declared order and is write-free for
    empty matrices.
    """

    def _validate_graph_and_schedule(
        self, plan: ResidentDiagnosticsPlan
    ) -> None:
        """Validate graph provenance, membership, and canonical order."""
        registry = cast(Any, plan.registry)
        if registry._session is not plan.session:
            raise ValueError("diagnostics registry must be bound to session.")
        registry.validate_pinned_session(plan.session)
        if not _is_resolver_produced_graph(plan.graph):
            raise ValueError(
                "diagnostics graph must be produced by plan resolution."
            )
        if not is_resolver_produced_schedule(plan.schedule, plan.graph):
            raise ValueError(
                "diagnostics schedule must be produced for the exact graph."
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

    def _validate_registrations(self, plan: ResidentDiagnosticsPlan) -> None:
        """Validate diagnostic registration ordering and uniqueness."""
        operations = tuple(item.operation for item in plan.registrations)
        if operations != tuple(ResidentDiagnosticOperation):
            raise ValueError(
                "diagnostic operations must be unique and match the closed "
                "canonical tuple."
            )

    def _validate_outputs(self, plan: ResidentDiagnosticsPlan) -> None:
        """Validate caller-owned diagnostic outputs against resident state."""
        registry = cast(Any, plan.registry)
        registry.validate_diagnostic_registrations(
            plan.session, plan.registrations
        )

    def _validate(self, plan: ResidentDiagnosticsPlan) -> None:
        """Validate plan provenance, closed operation order, and outputs.

        Args:
            plan: Exact diagnostics plan whose retained bindings are checked.

        Raises:
            ValueError: If lifecycle, graph, schedule, protocol, or output
                metadata validation fails.
        """
        self._validate_graph_and_schedule(plan)
        self._validate_registrations(plan)
        self._validate_outputs(plan)

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
        """Validate and dispatch each registration in declared order.

        Empty matrix schemas complete without their writer launch. Particle
        number still launches for ``(B, 0)`` because its ``(B,)`` output exists.

        Args:
            plan: Exact plan selecting the sources and caller-owned outputs.

        Raises:
            TypeError: If ``plan`` is not an exact diagnostics plan.
            ValueError: If its bindings or output metadata are invalid.
        """
        plan = self.validate(plan)
        dimensions = plan.session.dimensions
        if not dimensions.n_boxes:
            return
        particles = cast(Any, plan.session.particles)
        gas = cast(Any, plan.session.gas)
        environment = cast(Any, plan.session.environment)
        for registration in plan.registrations:
            operation = registration.operation
            if (
                operation
                is ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION
            ):
                wp.launch(
                    _particle_number,
                    dim=dimensions.n_boxes,
                    inputs=[particles.concentration, registration.output],
                    device=particles.concentration.device,
                )
            elif dimensions.n_species:
                matrix_dim = (dimensions.n_boxes, dimensions.n_species)
                if (
                    operation
                    is ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT
                ):
                    wp.launch(
                        _copy_snapshot,
                        dim=matrix_dim,
                        inputs=[gas.concentration, registration.output],
                        device=gas.concentration.device,
                    )
                elif (
                    operation
                    is ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT
                ):
                    wp.launch(
                        _copy_snapshot,
                        dim=matrix_dim,
                        inputs=[
                            environment.saturation_ratio,
                            registration.output,
                        ],
                        device=environment.saturation_ratio.device,
                    )
                elif (
                    operation is ResidentDiagnosticOperation.TOTAL_SPECIES_MASS
                ):
                    wp.launch(
                        _total_species_mass,
                        dim=matrix_dim,
                        inputs=[
                            particles.masses,
                            particles.concentration,
                            gas.concentration,
                            particles.volume,
                            registration.output,
                        ],
                        device=particles.masses.device,
                    )
                elif (
                    operation is ResidentDiagnosticOperation.LATENT_HEAT_ENERGY
                ):
                    wp.launch(
                        _copy_snapshot,
                        dim=matrix_dim,
                        inputs=[
                            registration.energy_transfer,
                            registration.output,
                        ],
                        device=particles.masses.device,
                    )
                else:
                    wp.launch(
                        _conservation_residual,
                        dim=matrix_dim,
                        inputs=[
                            particles.masses,
                            particles.concentration,
                            gas.concentration,
                            particles.volume,
                            registration.baseline_total_mass,
                            registration.source_ledger,
                            registration.sink_ledger,
                            registration.output,
                        ],
                        device=particles.masses.device,
                    )
