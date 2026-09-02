"""Write closed resident diagnostics into caller-owned Warp arrays.

This concrete direct-import-only module has no callback registration or package
export. Its shared validator performs read-only metadata checks without
constructing an executor or mutating resident state. Registrations execute in
this fixed order: gas-concentration snapshot,
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
write-free only for ``B == 0``. Prepared setup binds the closed plan; its
observation-free enqueue repeats no host/setup metadata validation and performs
no allocation, host readback, transfer, synchronization, lookup, or rebinding.
Retained diagnostic writers still perform device-side status and physical-state
validation. Setup rejection is pre-writer; rollback is not promised after a
diagnostic writer launches.
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
from particula.execution.resident_enqueue import PreparedResidentTimestep
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


@dataclass(frozen=True, eq=False)
class PreparedResidentDiagnostics:
    """Bind validated diagnostic sources and registrations for enqueue only.

    This concrete-only record retains every source, output, and operation by
    identity. Its observation-free enqueue dispatches those retained writers
    only; it does not revalidate or rediscover the plan.

    Attributes:
        plan: Exact diagnostics plan validated during setup.
        particle_masses: Bound resident particle mass array.
        particle_concentration: Bound resident particle concentration array.
        particle_volume: Bound resident particle volume array.
        gas_concentration: Bound resident gas concentration array.
        saturation_ratio: Bound resident saturation-ratio array.
        device: Device shared by bound arrays and outputs.
        dimensions: Exact resident dimensions used for empty-schema checks.
        registrations: Canonically ordered diagnostic registrations.
        outputs: Registration outputs in canonical launch order.
        energy_transfers: Latent-energy inputs aligned with registrations.
        baseline_total_masses: Residual baselines aligned with registrations.
        source_ledgers: Residual source ledgers aligned with registrations.
        sink_ledgers: Residual sink ledgers aligned with registrations.
        total_mass_output: Bound total-mass output used by the residual writer.
    """

    plan: ResidentDiagnosticsPlan
    particle_masses: object
    particle_concentration: object
    particle_volume: object
    gas_concentration: object
    saturation_ratio: object
    device: object
    dimensions: object
    registrations: tuple[ResidentDiagnosticRegistration, ...]
    outputs: tuple[object, ...]
    energy_transfers: tuple[object | None, ...]
    baseline_total_masses: tuple[object | None, ...]
    source_ledgers: tuple[object | None, ...]
    sink_ledgers: tuple[object | None, ...]
    total_mass_output: object | None


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
        return validate_resident_diagnostics_plan(plan)

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


def setup_prepared_resident_diagnostics(
    prepared_timestep: object, plan: object
) -> PreparedResidentDiagnostics:
    """Validate a P1 diagnostics attachment and bind enqueue identities.

    Args:
        prepared_timestep: Exact P1 timestep retaining ``plan`` by identity.
        plan: Exact resident diagnostics plan to validate and bind.

    Returns:
        Immutable binding that dispatches the canonical registrations without
        repeating setup validation.

    Raises:
        TypeError: If either carrier has an unsupported exact type.
        ValueError: If P1 identities, plan metadata, registrations, or primary
            arrays are invalid.
    """
    if type(prepared_timestep) is not PreparedResidentTimestep:
        raise TypeError(
            "prepared_timestep must be an exact PreparedResidentTimestep."
        )
    if type(plan) is not ResidentDiagnosticsPlan:
        raise TypeError("plan must be an exact ResidentDiagnosticsPlan.")
    prepared = cast(PreparedResidentTimestep, prepared_timestep)
    typed = cast(ResidentDiagnosticsPlan, plan)
    if prepared.request.diagnostics is not typed:
        raise ValueError("prepared timestep does not retain the supplied plan.")
    if (
        prepared.session is not typed.session
        or prepared.registry is not typed.registry
        or prepared.graph is not typed.graph
        or prepared.schedule is not typed.schedule
        or prepared.dimensions is not typed.session.dimensions
    ):
        raise ValueError("prepared timestep identities do not match plan.")
    validate_resident_diagnostics_plan(typed)
    particles = cast(Any, typed.session.particles)
    gas = cast(Any, typed.session.gas)
    environment = cast(Any, typed.session.environment)
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
        environment.temperature,
        environment.pressure,
        environment.saturation_ratio,
    )
    if len(prepared.primary_arrays) != len(primaries) or any(
        left is not right
        for left, right in zip(prepared.primary_arrays, primaries, strict=True)
    ):
        raise ValueError(
            "prepared timestep primary arrays do not match session."
        )
    total_mass_output = next(
        (
            registration.output
            for registration in typed.registrations
            if registration.operation
            is ResidentDiagnosticOperation.TOTAL_SPECIES_MASS
        ),
        None,
    )
    return PreparedResidentDiagnostics(
        typed,
        particles.masses,
        particles.concentration,
        particles.volume,
        gas.concentration,
        environment.saturation_ratio,
        particles.masses.device,
        typed.session.dimensions,
        typed.registrations,
        tuple(item.output for item in typed.registrations),
        tuple(item.energy_transfer for item in typed.registrations),
        tuple(item.baseline_total_mass for item in typed.registrations),
        tuple(item.source_ledger for item in typed.registrations),
        tuple(item.sink_ledger for item in typed.registrations),
        total_mass_output,
    )


def _enqueue_prepared_resident_diagnostics(  # noqa: C901
    prepared: PreparedResidentDiagnostics,
) -> None:
    """Dispatch only registrations retained by prepared diagnostics setup.

    This private retained-reference seam performs no validation, allocation,
    readback, transfer, synchronization, lookup, or rebinding. Empty matrix
    schemas remain write-free; rollback is not promised after a writer launches.

    Args:
        prepared: Previously validated immutable diagnostics binding.

    Raises:
        ValueError: If residual dispatch lacks its bound total-mass output.
    """
    dimensions = cast(Any, prepared.dimensions)
    device = cast(Any, prepared.device)
    if not dimensions.n_boxes:
        return
    for index, registration in enumerate(prepared.registrations):
        operation = registration.operation
        output = prepared.outputs[index]
        if (
            operation
            is ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION
        ):
            wp.launch(
                _clear_particle_number,
                dim=dimensions.n_boxes,
                inputs=[output],
                device=device,
            )
            if dimensions.n_particles:
                wp.launch(
                    _accumulate_particle_number,
                    dim=(dimensions.n_boxes, dimensions.n_particles),
                    inputs=[prepared.particle_concentration, output],
                    device=device,
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
                    inputs=[prepared.gas_concentration, output],
                    device=device,
                )
            elif (
                operation
                is ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT
            ):
                wp.launch(
                    _copy_snapshot,
                    dim=matrix_dim,
                    inputs=[prepared.saturation_ratio, output],
                    device=device,
                )
            elif operation is ResidentDiagnosticOperation.TOTAL_SPECIES_MASS:
                wp.launch(
                    _initialize_total_species_mass,
                    dim=matrix_dim,
                    inputs=[
                        prepared.gas_concentration,
                        prepared.particle_volume,
                        output,
                    ],
                    device=device,
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
                            prepared.particle_masses,
                            prepared.particle_concentration,
                            prepared.particle_volume,
                            output,
                        ],
                        device=device,
                    )
            elif operation is ResidentDiagnosticOperation.LATENT_HEAT_ENERGY:
                wp.launch(
                    _copy_snapshot,
                    dim=matrix_dim,
                    inputs=[prepared.energy_transfers[index], output],
                    device=device,
                )
            else:
                if prepared.total_mass_output is None:
                    raise ValueError(
                        "conservation residual requires total species mass."
                    )
                wp.launch(
                    _conservation_residual,
                    dim=matrix_dim,
                    inputs=[
                        prepared.total_mass_output,
                        prepared.baseline_total_masses[index],
                        prepared.source_ledgers[index],
                        prepared.sink_ledgers[index],
                        output,
                    ],
                    device=device,
                )


def validate_resident_diagnostics_plan(  # noqa: C901
    plan: object,
) -> ResidentDiagnosticsPlan:
    """Validate an exact diagnostics plan without constructing an executor.

    This read-only shared validation seam performs no diagnostic dispatch,
    resource acquisition, payload inspection, transfer, synchronization, or
    resident-state mutation.

    Args:
        plan: Candidate concrete diagnostics plan.

    Returns:
        The unchanged, exact validated plan.

    Raises:
        TypeError: If ``plan`` is not an exact diagnostics plan.
        ValueError: If its binding, graph, or registration metadata is invalid.
    """
    if type(plan) is not ResidentDiagnosticsPlan:
        raise TypeError("plan must be an exact ResidentDiagnosticsPlan.")
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
        or plan.node.resources
        != frozenset(
            {
                ResourceRequirement.PARTICLES,
                ResourceRequirement.GAS,
                ResourceRequirement.ENVIRONMENT,
                ResourceRequirement.THERMODYNAMICS,
                ResourceRequirement.DIAGNOSTICS,
            }
        )
    ):
        raise ValueError("diagnostics node has an invalid canonical role.")
    if plan.schedule.ordered_node_ids[-1:] != ("diagnostics",):
        raise ValueError("diagnostics must be the final scheduled node.")
    if plan.schedule.ordered_node_ids != resolve_canonical_topological_order(
        plan.schedule.nodes, plan.schedule.dependencies
    ):
        raise ValueError("diagnostics schedule must be canonical.")
    operations = tuple(item.operation for item in plan.registrations)
    if operations not in (
        tuple(ResidentDiagnosticOperation)[:2],
        tuple(ResidentDiagnosticOperation),
    ):
        raise ValueError(
            "diagnostic operations must be unique and match the legacy "
            "two-snapshot or current six-operation canonical tuple."
        )
    registry.validate_diagnostic_registrations(plan.session, plan.registrations)
    return plan
