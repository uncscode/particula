"""Write closed resident diagnostics into caller-owned Warp arrays.

This concrete direct-import-only module has no callback registration or package
export. Registrations execute in this fixed order: gas-concentration snapshot,
saturation-ratio snapshot, total species mass, particle-number concentration,
latent heat energy, and conservation residual. Matrix operations use ``(B, S)``
``wp.float64`` arrays; particle number uses a ``(B,)`` ``wp.float64`` array.

Total species mass is ``V[b] * (Σp(m[b, p, s] * c[b, p]) + g[b, s])`` in kg.
Particle number is ``Σp(c[b, p])`` in m^-3. Latent energy copies signed
whole-call P2-finalized energy in J. The residual is
``total_mass - baseline_total_mass - source_ledger + sink_ledger`` in kg;
source and sink ledgers are nonnegative accumulated extensive-mass inputs.
Execution validates caller-owned same-device bindings without host readback,
synchronization, transfer, allocation, or physics mutation. Empty matrix
operations are write-free for ``B == 0`` or ``S == 0``; particle number is
write-free only for ``B == 0``.
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
        operation: Exact closed operation that selects the diagnostic reduction.
        output: Caller-owned Warp ``float64`` output validated by the executor.
        energy_transfer: Required ``(B, S)`` signed whole-call energy input in
            J for latent-energy output; forbidden otherwise.
        baseline_total_mass: Required ``(B, S)`` extensive mass baseline in kg
            for residual output; forbidden otherwise.
        source_ledger: Required nonnegative extensive source ledger for the
            residual in kg; forbidden otherwise.
        sink_ledger: Required nonnegative extensive sink ledger for residual in
            kg; forbidden otherwise.
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
            ValueError: If required accounting inputs are missing or forbidden
                accounting inputs are supplied for ``operation``.
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
        registrations: Exact canonical tuple of the six ordered closed
            operation and output bindings, validated by the executor.
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
def _initialize_total_species_mass(
    gas_concentration: Any, volume: Any, output: Any
) -> None:
    """Initialize each extensive-mass lane from resident gas."""
    box, species = wp.tid()  # type: ignore[misc]
    output[box, species] = volume[box] * gas_concentration[box, species]


@wp.kernel
def _accumulate_particle_species_mass(
    masses: Any,
    concentration: Any,
    volume: Any,
    output: Any,
) -> None:
    """Accumulate one concentration-weighted particle lane in parallel."""
    box, particle, species = wp.tid()  # type: ignore[misc]
    wp.atomic_add(
        output,
        box,
        species,
        volume[box]
        * masses[box, particle, species]
        * concentration[box, particle],
    )


@wp.kernel
def _clear_particle_number(output: Any) -> None:
    """Clear one particle-number output lane before staged accumulation."""
    box = wp.tid()  # type: ignore[misc]
    output[box] = wp.float64(0.0)


@wp.kernel
def _accumulate_particle_number(concentration: Any, output: Any) -> None:
    """Accumulate one particle concentration lane in parallel."""
    box, particle = wp.tid()  # type: ignore[misc]
    wp.atomic_add(output, box, concentration[box, particle])


@wp.kernel
def _conservation_residual(
    total_mass: Any,
    baseline: Any,
    source: Any,
    sink: Any,
    output: Any,
) -> None:
    """Write the ledger-aware residual from the already-reduced total mass."""
    box, species = wp.tid()  # type: ignore[misc]
    output[box, species] = (
        total_mass[box, species]
        - baseline[box, species]
        - source[box, species]
        + sink[box, species]
    )


class ResidentDiagnosticsExecutor:
    """Execute an already-bound closed diagnostics plan without transfers.

    Validation preserves caller ownership and rejects output or accounting-input
    aliases with resident primaries, published sidecars, or diagnostic outputs.
    Execution dispatches the six canonical registrations without host readback,
    synchronization, transfer, allocation, or physics mutation. Matrix
    registrations are write-free for empty ``(B, S)`` schemas; particle number
    remains writable for ``(B, 0)``.
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
        legacy_operations = tuple(ResidentDiagnosticOperation)[:2]
        if operations not in (
            legacy_operations,
            tuple(ResidentDiagnosticOperation),
        ):
            raise ValueError(
                "diagnostic operations must be unique and match the legacy "
                "two-snapshot or current six-operation canonical tuple."
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

        Args:
            plan: Candidate concrete diagnostics plan.

        Returns:
            The unchanged, exact validated plan.

        Raises:
            TypeError: If ``plan`` is not an exact diagnostics plan.
            ValueError: If the plan's graph, bindings, or registration protocol
                is invalid.
        """
        if type(plan) is not ResidentDiagnosticsPlan:
            raise TypeError("plan must be an exact ResidentDiagnosticsPlan.")
        self._validate(plan)
        return plan

    def execute(self, plan: object) -> None:
        """Validate and dispatch each registration in declared order.

        Matrix schemas complete without their writer launch when ``B == 0`` or
        ``S == 0``. Particle number still launches for ``(B, 0)`` because its
        ``(B,)`` output exists. Successful launches are asynchronous; callers
        synchronize before inspecting outputs on the host.

        Args:
            plan: Exact plan selecting the sources and caller-owned outputs.

        Raises:
            TypeError: If ``plan`` is not an exact diagnostics plan.
            ValueError: If its bindings or output metadata are invalid.
        """
        plan = self.validate(plan)
        self._execute_validated(plan)

    def _execute_validated(  # noqa: C901
        self, plan: ResidentDiagnosticsPlan
    ) -> None:
        """Dispatch a plan already validated by its owning scheduler step."""
        dimensions = plan.session.dimensions
        if not dimensions.n_boxes:
            return
        particles = cast(Any, plan.session.particles)
        gas = cast(Any, plan.session.gas)
        environment = cast(Any, plan.session.environment)
        total_mass_output = next(
            (
                registration.output
                for registration in plan.registrations
                if registration.operation
                is ResidentDiagnosticOperation.TOTAL_SPECIES_MASS
            ),
            None,
        )
        for registration in plan.registrations:
            operation = registration.operation
            if (
                operation
                is ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION
            ):
                wp.launch(
                    _clear_particle_number,
                    dim=dimensions.n_boxes,
                    inputs=[registration.output],
                    device=particles.concentration.device,
                )
                if dimensions.n_particles:
                    wp.launch(
                        _accumulate_particle_number,
                        dim=(dimensions.n_boxes, dimensions.n_particles),
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
                        _initialize_total_species_mass,
                        dim=matrix_dim,
                        inputs=[
                            gas.concentration,
                            particles.volume,
                            registration.output,
                        ],
                        device=particles.masses.device,
                    )
                    if dimensions.n_particles:
                        wp.launch(
                            _accumulate_particle_species_mass,
                            dim=(
                                dimensions.n_boxes,
                                dimensions.n_particles,
                                dimensions.n_species,
                            ),
                            inputs=[
                                particles.masses,
                                particles.concentration,
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
                    if total_mass_output is None:
                        raise ValueError(
                            "conservation residual requires total species mass."
                        )
                    wp.launch(
                        _conservation_residual,
                        dim=matrix_dim,
                        inputs=[
                            total_mass_output,
                            registration.baseline_total_mass,
                            registration.source_ledger,
                            registration.sink_ledger,
                            registration.output,
                        ],
                        device=particles.masses.device,
                    )
