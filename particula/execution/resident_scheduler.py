"""Run the closed twelve-node GPU-resident simulation schedule.

This concrete direct-import-only composition boundary dispatches communication,
then optional volume evolution, before the ten ordinary loop nodes. It retains
every resident object by identity and performs no upload, restore,
synchronization, fallback, resource acquisition, retry, or rollback.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

from particula.execution import _isfinite_real
from particula.execution.adapters.coagulation import (
    WarpBrownianCoagulationExecutionAdapter,
    WarpBrownianCoagulationExecutionState,
)
from particula.execution.adapters.condensation import (
    WarpCondensationExecutionAdapter,
    WarpCondensationExecutionState,
)
from particula.execution.diagnostics import (
    ResidentDiagnosticsExecutor,
    ResidentDiagnosticsPlan,
)
from particula.execution.gpu_session import (
    ResidentSession,
    ResidentStepGuard,
    _handle_failed_resident_operation,
    _ResidentOperationOutcome,
)
from particula.execution.process_adapters import (
    ResidentDilutionAdapter,
    ResidentDilutionRequest,
    ResidentNucleationAdapter,
    ResidentNucleationRequest,
    ResidentWallLossAdapter,
    ResidentWallLossRequest,
)
from particula.execution.process_graph import (
    DependencyEdge,
    ProcessNode,
    ResolvedProcessGraph,
    _is_resolver_produced_graph,
    resolve_canonical_topological_order,
)
from particula.execution.resident_communication import (
    ResidentCommunicationExecutor,
    ResidentCommunicationRequest,
)
from particula.execution.scheduler import (
    ResolvedTimestepSchedule,
    is_resolver_produced_schedule,
)
from particula.execution.state_updates import (
    ResidentEnvironmentUpdateRequest,
    ResidentGasUpdateRequest,
    ResidentStateUpdateExecutor,
)
from particula.execution.thermodynamic_updates import (
    ResidentThermodynamicUpdateCoordinator,
    ResidentThermodynamicUpdateRequest,
)
from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig

_COMPLETE_IDS = frozenset(
    {
        "communication",
        "volume_evolution",
        "environment_update",
        "gas_update",
        "vapor_pressure_refresh",
        "saturation_refresh",
        "condensation",
        "brownian_coagulation",
        "dilution",
        "wall_loss",
        "nucleation",
        "diagnostics",
    }
)
_VIRTUAL_IDS = frozenset({"vapor_pressure_refresh", "saturation_refresh"})


def _registry_type() -> type[object]:
    """Return the concrete registry type without creating an import cycle.

    Returns:
        The direct-module-only GPU resource registry type.
    """
    from particula.execution.gpu_resources import GPUResourceRegistry

    return GPUResourceRegistry


@dataclass(frozen=True, eq=False)
class ResidentSimulationRequest:
    """Bind one complete resolved simulation loop to resident resources.

    The request retains all carriers by identity for the canonical twelve-node
    graph. It is concrete-only and does not acquire resources, begin a step,
    transfer, synchronize, or validate physical process inputs.

    Attributes:
        session: Exact active resident session.
        registry: Exact registry pinned to ``session``.
        guard: Exact closed lifecycle guard for the same binding.
        graph: Resolver-produced process graph for the complete loop.
        schedule: Canonical resolved schedule for ``graph``.
        thermodynamics: Exact configuration shared by thermal consumers.
        condensation: Exact resident condensation execution state.
        coagulation: Exact resident Brownian-coagulation execution state.
        dilution: Exact resident dilution request.
        wall_loss: Exact resident wall-loss request.
        nucleation: Exact resident nucleation request.
        diagnostics: Exact closed diagnostics plan.
        environment_update: Optional exact environment update request.
        gas_update: Optional exact gas update request.
        communication: Exact request for the communication and volume barriers.
    """

    session: ResidentSession
    registry: object
    guard: ResidentStepGuard
    graph: ResolvedProcessGraph
    schedule: ResolvedTimestepSchedule
    thermodynamics: ThermodynamicsConfig
    condensation: WarpCondensationExecutionState
    coagulation: WarpBrownianCoagulationExecutionState
    dilution: ResidentDilutionRequest
    wall_loss: ResidentWallLossRequest
    nucleation: ResidentNucleationRequest
    diagnostics: ResidentDiagnosticsPlan
    environment_update: ResidentEnvironmentUpdateRequest | None = None
    gas_update: ResidentGasUpdateRequest | None = None
    communication: ResidentCommunicationRequest | None = None

    def __post_init__(self) -> None:
        """Validate exact request components and optional update types.

        Raises:
            TypeError: If a required component or optional update has an
                inexact concrete type.
        """
        exact = (
            (self.session, ResidentSession, "session"),
            (self.registry, _registry_type(), "registry"),
            (self.guard, ResidentStepGuard, "guard"),
            (self.graph, ResolvedProcessGraph, "graph"),
            (self.schedule, ResolvedTimestepSchedule, "schedule"),
            (self.thermodynamics, ThermodynamicsConfig, "thermodynamics"),
            (self.condensation, WarpCondensationExecutionState, "condensation"),
            (
                self.coagulation,
                WarpBrownianCoagulationExecutionState,
                "coagulation",
            ),
            (self.dilution, ResidentDilutionRequest, "dilution"),
            (self.wall_loss, ResidentWallLossRequest, "wall_loss"),
            (self.nucleation, ResidentNucleationRequest, "nucleation"),
            (self.diagnostics, ResidentDiagnosticsPlan, "diagnostics"),
        )
        for value, expected, name in exact:
            if type(value) is not expected:
                raise TypeError(f"{name} must be an exact {expected.__name__}.")
        if (
            self.environment_update is not None
            and type(self.environment_update)
            is not ResidentEnvironmentUpdateRequest
        ):
            raise TypeError(
                "environment_update must be an exact request or None."
            )
        if (
            self.gas_update is not None
            and type(self.gas_update) is not ResidentGasUpdateRequest
        ):
            raise TypeError("gas_update must be an exact request or None.")
        if (
            self.communication is not None
            and type(self.communication) is not ResidentCommunicationRequest
        ):
            raise TypeError("communication must be an exact request or None.")


class ResidentSimulationScheduler:
    """Execute one canonical fully resolved resident timestep at a time.

    Each successful call opens and completes exactly one lifecycle token while
    dispatching the resolved twelve-node schedule. Communication runs with
    pre-update volumes, optional volume evolution follows it, and both barriers
    invalidate saturation ratio only. The scheduler neither transfers nor
    restores data, acquires resources, synchronizes, retries, falls back, or
    rolls back after a writer-capable operation may have launched.
    """

    def __init__(self, request: ResidentSimulationRequest) -> None:
        """Retain one exact resident simulation request.

        Args:
            request: Complete identity-bound resident simulation request.

        Raises:
            TypeError: If ``request`` is not an exact request instance.
        """
        if type(request) is not ResidentSimulationRequest:
            raise TypeError(
                "request must be an exact ResidentSimulationRequest."
            )
        self._request = request

    def _validate(self, duration: object) -> None:  # noqa: C901
        """Preflight the lifecycle, graph, request, and duration bindings.

        Args:
            duration: Candidate nonnegative finite timestep duration.

        Raises:
            TypeError: If the duration is not a non-boolean real value.
            ValueError: If duration, ownership, graph, schedule, request, or
                diagnostics validation fails.
        """
        request = self._request
        registry = cast(Any, request.registry)
        if isinstance(duration, bool) or not isinstance(duration, Real):
            raise TypeError("duration must be a non-boolean real.")
        if not _isfinite_real(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative.")
        if (
            registry._session is not request.session
            or request.guard._session is not request.session
            or request.guard._registry is not request.registry
        ):
            raise ValueError(
                "guard must match the resident session and registry."
            )
        request.guard.assert_step_closed()
        registry.validate_pinned_session(request.session)
        if not _is_resolver_produced_graph(request.graph):
            raise ValueError("graph must be produced by plan resolution.")
        if not is_resolver_produced_schedule(request.schedule, request.graph):
            raise ValueError(
                "schedule must be produced for the exact resolved graph."
            )
        ids = request.schedule.ordered_node_ids
        complete_ids = frozenset(ids)
        if complete_ids != _COMPLETE_IDS or len(ids) != len(complete_ids):
            raise ValueError(
                "schedule must contain exactly the complete resident loop."
            )
        if request.communication is None:
            raise ValueError(
                "complete barrier schedule requires communication request."
            )
        if ids != resolve_canonical_topological_order(
            request.schedule.nodes, request.schedule.dependencies
        ):
            raise ValueError("schedule must use canonical topological order.")
        graph_by_id = {node.node_id: node for node in request.graph.nodes}
        for node in request.schedule.nodes:
            node_id = node.node_id
            if graph_by_id.get(node_id) is not node:
                raise ValueError(
                    "schedule nodes must be identical graph members."
                )
        self._validate_virtual_refresh_windows(
            ids, request.schedule.dependencies
        )
        self._validate_request_nodes(graph_by_id)
        ResidentDiagnosticsExecutor().validate(request.diagnostics)
        self._validate_durations(duration)

    @staticmethod
    def _validate_virtual_refresh_windows(
        ids: tuple[str, ...], dependencies: tuple[DependencyEdge, ...]
    ) -> None:
        """Require the complete resolver freshness window.

        The refresh nodes must remain adjacent to condensation and diagnostics
        before lifecycle entry.
        """
        positions = {node_id: index for index, node_id in enumerate(ids)}
        vapor = positions["vapor_pressure_refresh"]
        saturation = positions["saturation_refresh"]
        condensation = positions["condensation"]
        diagnostics = positions["diagnostics"]
        pairs = {(edge.before_id, edge.after_id) for edge in dependencies}
        if (
            saturation != vapor + 1
            or condensation != saturation + 1
            or diagnostics != len(ids) - 1
            or ("vapor_pressure_refresh", "saturation_refresh") not in pairs
            or ("saturation_refresh", "condensation") not in pairs
            or ("saturation_refresh", "diagnostics") not in pairs
        ):
            raise ValueError(
                "schedule must retain complete thermodynamic refresh windows."
            )

    def _validate_request_nodes(  # noqa: C901
        self, graph_by_id: dict[str, ProcessNode]
    ) -> None:
        """Validate request bindings against exact resolved graph nodes.

        Args:
            graph_by_id: Resolved graph nodes indexed by their canonical IDs.

        Raises:
            ValueError: If a state update, diagnostics plan, process request, or
            execution state does not retain the scheduler's binding. This
            metadata preflight does not dispatch, transfer, synchronize, acquire
            resources, or mutate resident arrays.
        """
        request = self._request
        registry = cast(Any, request.registry)
        node_fields = (
            (request.environment_update, "environment_update"),
            (request.gas_update, "gas_update"),
        )
        for item, node_id in node_fields:
            if (
                item is None
                or item.graph is not request.graph
                or item.session is not request.session
                or item.registry is not request.registry
                or item.node is not graph_by_id[node_id]
            ):
                raise ValueError(
                    "state update request does not match resolved binding."
                )
        plan = request.diagnostics
        if (
            plan.session is not request.session
            or plan.registry is not request.registry
            or plan.graph is not request.graph
            or plan.schedule is not request.schedule
            or plan.node is not graph_by_id["diagnostics"]
        ):
            raise ValueError(
                "diagnostics plan does not match resolved binding."
            )
        for process_request in (
            request.dilution,
            request.wall_loss,
            request.nucleation,
        ):
            if (
                process_request.session is not request.session
                or process_request.registry is not request.registry
            ):
                raise ValueError(
                    "process request does not match resident binding."
                )
        communication = request.communication
        if communication is not None:
            if (
                communication.session is not request.session
                or communication.registry is not request.registry
                or communication.graph is not request.graph
                or communication.duration != request.dilution.time_step
                or communication.communication_node
                is not graph_by_id.get("communication")
                or communication.volume_evolution_node
                is not graph_by_id.get("volume_evolution")
            ):
                raise ValueError(
                    "communication request does not match resolved binding."
                )
            ResidentCommunicationExecutor(communication).validate()
        condensation = request.condensation.state
        if (
            condensation.particles is not request.session.particles
            or condensation.gas is not request.session.gas
            or condensation.environment is not request.session.environment
            or condensation.thermodynamics is not request.thermodynamics
        ):
            raise ValueError(
                "condensation state does not match resident binding."
            )
        coagulation = request.coagulation.state
        if (
            coagulation.particles is not request.session.particles
            or coagulation.environment is not request.session.environment
        ):
            raise ValueError(
                "coagulation state does not match resident binding."
            )
        condensation_resources = registry._views.get("condensation")
        if condensation.scratch_buffers is not getattr(
            condensation_resources, "scratch_buffers", None
        ):
            raise ValueError("condensation state must use published resources.")
        registry.validate_condensation_resources(
            request.session, condensation_resources
        )
        coagulation_resources = registry._views.get("coagulation")
        if (
            coagulation.collision_pairs
            is not getattr(coagulation_resources, "collision_pairs", None)
            or coagulation.n_collisions
            is not getattr(coagulation_resources, "n_collisions", None)
            or coagulation.rng_states
            is not getattr(coagulation_resources, "rng_states", None)
        ):
            raise ValueError("coagulation state must use published resources.")
        registry.validate_coagulation_resources(
            request.session, coagulation_resources
        )
        registry.validate_wall_loss_resources(
            request.session, request.wall_loss.resources
        )
        registry.validate_nucleation_resources(
            request.session, request.nucleation.resources
        )

    def _validate_durations(self, duration: Real) -> None:
        """Require every process request to retain the exact step duration.

        Args:
            duration: Prevalidated duration supplied to :meth:`execute`.

        Raises:
            ValueError: If a participating process ``time_step`` differs.
        """
        request = self._request
        values = (
            request.condensation.time_step,
            request.coagulation.state.time_step,
            request.dilution.time_step,
            request.wall_loss.time_step,
            request.nucleation.time_step,
        )
        if any(value != duration for value in values):
            raise ValueError(
                "all process time_step values must equal duration."
            )

    def execute(self, duration: object) -> None:  # noqa: C901
        """Preflight then run one complete ordered timestep.

        Failures before a writer-capable invocation leave the session active.
        Once dispatch begins, the token is closed and the session faults without
        rollback because a native writer may already have launched.

        Args:
            duration: Nonnegative finite duration matching each process request.

        Raises:
            TypeError: If preflight finds an inexact carrier or invalid
                duration.
            ValueError: If resolved bindings, duration agreement, or an invoked
                operation reject execution.
            RuntimeError: If lifecycle token handling rejects the timestep.
        """
        self._validate(duration)
        request = self._request
        updates = ResidentStateUpdateExecutor()
        thermal = ResidentThermodynamicUpdateCoordinator(
            ResidentThermodynamicUpdateRequest(
                request.session,
                request.registry,
                request.graph,
                request.schedule,
                request.thermodynamics,
            )
        )
        condensation = WarpCondensationExecutionAdapter()
        coagulation = WarpBrownianCoagulationExecutionAdapter()
        dilution = ResidentDilutionAdapter()
        wall_loss = ResidentWallLossAdapter()
        nucleation = ResidentNucleationAdapter()
        diagnostics = ResidentDiagnosticsExecutor()
        communication = (
            None
            if request.communication is None
            else ResidentCommunicationExecutor(request.communication)
        )
        token = request.guard.begin_step(duration)
        writer_called = False
        try:
            graph_by_id = {node.node_id: node for node in request.graph.nodes}
            for node_id in request.schedule.ordered_node_ids:
                node = graph_by_id[node_id]
                if node_id in _VIRTUAL_IDS:
                    continue
                writer_called = True
                if node_id == "communication":
                    if communication is None:
                        raise ValueError("communication request is required.")
                    communication.execute_communication()
                    thermal.record_completed(node)
                elif node_id == "volume_evolution":
                    if communication is None:
                        raise ValueError("communication request is required.")
                    communication.execute_volume_evolution()
                    thermal.record_completed(node)
                elif node_id == "environment_update":
                    updates.execute(request.environment_update)
                    thermal.record_completed(node)
                elif node_id == "gas_update":
                    updates.execute(request.gas_update)
                    thermal.record_completed(node)
                elif node_id == "condensation":
                    thermal.execute_consumer(
                        node, lambda: condensation.execute(request.condensation)
                    )
                elif node_id == "brownian_coagulation":
                    coagulation.execute(request.coagulation)
                    thermal.record_completed(node)
                elif node_id == "dilution":
                    dilution.execute(request.dilution)
                    thermal.record_completed(node)
                elif node_id == "wall_loss":
                    wall_loss.execute(request.wall_loss)
                    thermal.record_completed(node)
                elif node_id == "nucleation":
                    thermal.execute_consumer(
                        node, lambda: nucleation.execute(request.nucleation)
                    )
                elif node_id == "diagnostics":
                    thermal.execute_consumer(
                        node, lambda: diagnostics.execute(request.diagnostics)
                    )
            writer_called = True
            request.guard.complete_step(token)
        except BaseException as error:
            outcome = (
                _ResidentOperationOutcome.WRITER_MAY_HAVE_LAUNCHED
                if writer_called
                else _ResidentOperationOutcome.READ_ONLY
            )
            try:
                _handle_failed_resident_operation(
                    request.session,
                    cast(Any, request.registry),
                    request.guard,
                    token,
                    outcome,
                )
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
