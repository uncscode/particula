"""Run legacy and READY-prepared twelve-node GPU-resident schedules.

This concrete direct-import-only composition boundary dispatches communication,
then optional volume evolution, before the ten ordinary loop nodes. It retains
every resident object by identity and performs no upload, restore,
synchronization, fallback, resource acquisition, retry, or rollback.
Legacy scheduler execution gates an attached graph-capture binding by its exact
resident identities, capability, lifecycle, cached capture-resource publication,
and structural signature before the step token is opened. The prepared path is
concrete-only, READY, and uncaptured; it retains setup-validated operation
callables and rechecks its attachment, cached capture-resource publication, and
structural signature before opening its token. After this mandatory pre-token
gate, prepared dispatch repeats no host/setup metadata validation and performs
no host readback, allocation, transfer, synchronization, lookup, rebinding, or
RNG reset. Retained native operations still perform their device-side status and
physical-state validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, cast

from particula.execution import _isfinite_real
from particula.execution.adapters.coagulation import (
    ResidentBrownianCoagulationExecutionAdapter,
    ResidentBrownianCoagulationExecutionState,
    WarpBrownianCoagulationExecutionAdapter,  # noqa: F401
)
from particula.execution.adapters.condensation import (
    WarpCondensationExecutionAdapter,
    WarpCondensationExecutionState,
)
from particula.execution.diagnostics import (
    ResidentDiagnosticsExecutor,
    ResidentDiagnosticsPlan,
    _enqueue_prepared_resident_diagnostics,
    setup_prepared_resident_diagnostics,
    validate_resident_diagnostics_plan,
)
from particula.execution.gpu_session import (
    ResidentSession,
    ResidentStepGuard,
    ResidentStepToken,
    _handle_failed_resident_operation,
    _ResidentOperationOutcome,
)
from particula.execution.graph_capture import (
    _fault_resident_graph_capture_after_classification_failure,
    classify_resident_graph_capture_writer_failure,
    compare_resident_graph_capture_signature,
    gate_resident_graph_capture,
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

if TYPE_CHECKING:
    from particula.execution.gpu_resources import (
        CaptureResourceRequirements,
        GPUResourceRegistry,
    )
from particula.execution.resident_communication import (
    PreparedResidentCommunicationBinding,
    ResidentCommunicationExecutor,
    ResidentCommunicationRequest,
    _enqueue_prepared_resident_communication_node,
    _enqueue_prepared_resident_volume_evolution_node,
    setup_prepared_resident_communication,
    validate_resident_communication_request,
)
from particula.execution.resident_enqueue import (
    PreparedResidentTimestep,
    _validate_ready_attachment,
    prepare_resident_timestep,
)
from particula.execution.scheduler import (
    ResolvedTimestepSchedule,
    is_resolver_produced_schedule,
)
from particula.execution.state_updates import (
    ResidentEnvironmentUpdateRequest,
    ResidentGasUpdateRequest,
    ResidentStateUpdateExecutor,
    _enqueue_prepared_environment_update,
    _enqueue_prepared_gas_update,
    setup_prepared_environment_update,
    setup_prepared_gas_update,
)
from particula.execution.thermodynamic_updates import (
    PreparedResidentThermodynamicSequence,
    ResidentThermodynamicUpdateCoordinator,
    ResidentThermodynamicUpdateRequest,
    _enqueue_prepared_saturation_ratio,
    _enqueue_prepared_vapor_pressure,
    record_prepared_thermodynamic_success,
    setup_prepared_thermodynamic_sequence,
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
_CANONICAL_IDS = (
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
)
_CANONICAL_WRITER_CAPABLE = (True,) * len(_CANONICAL_IDS)


@dataclass(frozen=True, eq=False)
class PreparedResidentOperation:
    """Retain one setup-validated resident operation for direct dispatch.

    Attributes:
        node: Resolver-produced node represented by this operation.
        handler: Scheduler-owned fixed handler for the prepared operation.
        arguments: Setup-bound arguments supplied to ``handler`` at dispatch.
        product: Setup product retained by identity for drift detection.
        writer_capable: Whether invoking ``handler`` may launch a device writer.
    """

    node: ProcessNode
    handler: Callable[..., object]
    arguments: tuple[object, ...]
    product: object
    writer_capable: bool


@dataclass(frozen=True, eq=False)
class PreparedResidentSimulation:
    """Freeze all READY-bound products required for one uncaptured timestep.

    The carrier is identity-semantic: its metadata, setup products, node order,
    and operation callables are all retained during preparation. Enqueue can
    therefore dispatch the complete schedule without reconstructing executors,
    resolving nodes, repeating payload validation, or invoking graph-capture
    admission. It represents a READY attachment, not a captured graph.

    Attributes:
        timestep: Prepared P1 timestep retained by identity.
        request: Complete resident request retained by identity.
        session: Resident session owning the dispatched containers.
        registry: Resource registry pinned to ``session``.
        guard: Closed step guard for the exact session and registry binding.
        lifecycle: READY lifecycle metadata retained from the P1 timestep.
        signature: Prepared graph-capture signature metadata.
        graph: Resolver-produced graph for the complete loop.
        schedule: Resolver-produced schedule for the complete loop.
        ordered_node_ids: Canonical twelve-node operation identifiers.
        primary_arrays: Resident primary arrays retained by identity.
        resource_views: Published resource views retained by identity.
        capture_requirements: Pre-published capture requirements retained by
            identity.
        capture_set: Registry-published capture resource set retained by
            identity.
        capture_report: Cached immutable logical-byte report retained by
            identity.
        nodes: Canonical graph nodes in operation order.
        thermal: Prepared thermodynamic refresh sequence.
        communication: Prepared communication and volume barriers.
        environment: Prepared environment update operation.
        gas: Prepared gas update operation.
        condensation: Prepared condensation operation.
        coagulation: Prepared Brownian coagulation operation.
        dilution: Prepared dilution operation.
        wall_loss: Prepared wall-loss operation.
        nucleation: Prepared nucleation operation.
        diagnostics: Prepared diagnostics operation.
        operations: Ordered callables dispatched during enqueue.
        duration: Nonnegative timestep duration in seconds.
    """

    timestep: PreparedResidentTimestep
    request: ResidentSimulationRequest
    session: object
    registry: GPUResourceRegistry
    guard: object
    lifecycle: object
    signature: object
    graph: object
    schedule: object
    ordered_node_ids: tuple[object, ...]
    primary_arrays: tuple[object, ...]
    resource_views: tuple[object, ...]
    capture_requirements: object
    capture_set: object
    capture_report: object
    nodes: tuple[ProcessNode, ...]
    thermal: PreparedResidentThermodynamicSequence
    communication: PreparedResidentCommunicationBinding
    environment: object
    gas: object
    condensation: object
    coagulation: object
    dilution: object
    wall_loss: object
    nucleation: object
    diagnostics: object
    operations: tuple[PreparedResidentOperation, ...]
    duration: object


def _registry_type() -> type[object]:
    """Return the concrete registry type without creating an import cycle.

    Returns:
        The direct-module-only GPU resource registry type.
    """
    from particula.execution.gpu_resources import GPUResourceRegistry

    return GPUResourceRegistry


def _graph_capture_binding_type() -> type[object]:
    """Return the concrete optional graph-capture binding type lazily."""
    from particula.execution.graph_capture import ResidentGraphCaptureBinding

    return ResidentGraphCaptureBinding


def _capture_resource_requirements_type() -> type[object]:
    """Return the concrete capture requirements type without an import cycle."""
    from particula.execution.gpu_resources import CaptureResourceRequirements

    return CaptureResourceRequirements


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
        coagulation: Resident Brownian execution state binding the published
            coagulation-only RNG sidecar by identity and requiring forced-false
            RNG initialization during dispatch.
        dilution: Exact resident dilution request.
        wall_loss: Exact resident wall-loss request retaining the published
            wall-loss RNG sidecar by identity. Its reset flag must be literal
            ``False`` and its scheduler-owned selected logical-box indices must
            be validated in ascending order before dispatch.
        nucleation: Exact resident nucleation request.
        diagnostics: Exact closed diagnostics plan.
        environment_update: Optional exact environment update request.
        gas_update: Optional exact gas update request.
        communication: Exact request for the communication and volume barriers.
        capture_resource_requirements: Exact pre-published capture resource
            requirements retained for metadata-only admission checks.
        graph_capture_binding: Optional exact direct-module-only binding. It is
            attached only after final request construction and gates admission;
            it neither captures nor replays graphs nor changes dispatch.
    """

    session: ResidentSession
    registry: GPUResourceRegistry
    guard: ResidentStepGuard
    graph: ResolvedProcessGraph
    schedule: ResolvedTimestepSchedule
    thermodynamics: ThermodynamicsConfig
    condensation: WarpCondensationExecutionState
    coagulation: ResidentBrownianCoagulationExecutionState
    dilution: ResidentDilutionRequest
    wall_loss: ResidentWallLossRequest
    nucleation: ResidentNucleationRequest
    diagnostics: ResidentDiagnosticsPlan
    environment_update: ResidentEnvironmentUpdateRequest | None
    gas_update: ResidentGasUpdateRequest | None
    communication: ResidentCommunicationRequest | None
    capture_resource_requirements: CaptureResourceRequirements
    graph_capture_binding: object | None = None

    def __post_init__(self) -> None:
        """Validate exact request components and optional update types.

        Raises:
            TypeError: If a required component or optional update has an
                inexact concrete type.
        """
        if (
            type(self.coagulation)
            is not ResidentBrownianCoagulationExecutionState
        ):
            raise TypeError(
                "coagulation must be an exact resident execution state."
            )
        exact = (
            (self.session, ResidentSession, "session"),
            (self.registry, _registry_type(), "registry"),
            (self.guard, ResidentStepGuard, "guard"),
            (self.graph, ResolvedProcessGraph, "graph"),
            (self.schedule, ResolvedTimestepSchedule, "schedule"),
            (self.thermodynamics, ThermodynamicsConfig, "thermodynamics"),
            (self.condensation, WarpCondensationExecutionState, "condensation"),
            (self.dilution, ResidentDilutionRequest, "dilution"),
            (self.wall_loss, ResidentWallLossRequest, "wall_loss"),
            (self.nucleation, ResidentNucleationRequest, "nucleation"),
            (self.diagnostics, ResidentDiagnosticsPlan, "diagnostics"),
            (
                self.capture_resource_requirements,
                _capture_resource_requirements_type(),
                "capture_resource_requirements",
            ),
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
        if (
            self.graph_capture_binding is not None
            and type(self.graph_capture_binding)
            is not _graph_capture_binding_type()
        ):
            raise TypeError(
                "graph_capture_binding must be an exact binding or None."
            )


def prepare_resident_simulation(
    request: object, duration: object
) -> PreparedResidentSimulation:
    """Prepare all twelve resident operations while the attachment is READY.

    Setup deliberately performs validation and private kernel preparation before
    the returned carrier exists. The later enqueue path only invokes the
    retained callables under one lifecycle token.

    Args:
        request: Exact complete resident simulation request to prepare.
        duration: Finite, nonnegative timestep duration in seconds.

    Returns:
        Frozen identity-semantic carrier containing the twelve prepared
        operations.

    Raises:
        TypeError: If a request or prepared component has an invalid type.
        ValueError: If resident metadata, schedule, or resource identities do
            not satisfy the READY preparation contract.
    """
    prepared = prepare_resident_timestep(request, duration)
    typed = prepared.request
    nodes = {node.node_id: node for node in typed.schedule.nodes}
    communication = setup_prepared_resident_communication(
        prepared, typed.communication
    )
    environment = setup_prepared_environment_update(
        prepared, typed.environment_update
    )
    gas = setup_prepared_gas_update(prepared, typed.gas_update)
    thermal = setup_prepared_thermodynamic_sequence(
        prepared,
        ResidentThermodynamicUpdateRequest(
            typed.session,
            typed.registry,
            typed.graph,
            typed.schedule,
            typed.thermodynamics,
        ),
        nodes["condensation"],
        nodes["diagnostics"],
    )
    condensation = WarpCondensationExecutionAdapter().prepare(
        typed.condensation
    )
    coagulation = ResidentBrownianCoagulationExecutionAdapter().prepare(
        typed.coagulation
    )
    dilution = ResidentDilutionAdapter().prepare(typed.dilution)
    wall_loss = ResidentWallLossAdapter().prepare(typed.wall_loss)
    nucleation = ResidentNucleationAdapter().prepare(typed.nucleation)
    diagnostics = setup_prepared_resident_diagnostics(
        prepared, typed.diagnostics
    )
    node_sequence = tuple(nodes[node_id] for node_id in _CANONICAL_IDS)
    operations = (
        PreparedResidentOperation(
            nodes["communication"],
            _enqueue_prepared_resident_communication_node,
            (communication,),
            communication,
            True,
        ),
        PreparedResidentOperation(
            nodes["volume_evolution"],
            _enqueue_prepared_resident_volume_evolution_node,
            (communication,),
            communication,
            True,
        ),
        PreparedResidentOperation(
            nodes["environment_update"],
            _enqueue_prepared_environment_update,
            (environment,),
            environment,
            True,
        ),
        PreparedResidentOperation(
            nodes["gas_update"],
            _enqueue_prepared_gas_update,
            (gas,),
            gas,
            True,
        ),
        PreparedResidentOperation(
            nodes["vapor_pressure_refresh"],
            _enqueue_prepared_vapor_pressure,
            (thermal.condensation,),
            thermal.condensation,
            True,
        ),
        PreparedResidentOperation(
            nodes["saturation_refresh"],
            _enqueue_prepared_saturation_ratio,
            (thermal.condensation,),
            thermal.condensation,
            True,
        ),
        PreparedResidentOperation(
            nodes["condensation"],
            _enqueue_prepared_operation_execute,
            (condensation,),
            condensation,
            True,
        ),
        PreparedResidentOperation(
            nodes["brownian_coagulation"],
            _enqueue_prepared_operation_execute,
            (coagulation,),
            coagulation,
            True,
        ),
        PreparedResidentOperation(
            nodes["dilution"],
            _enqueue_prepared_operation_execute,
            (dilution,),
            dilution,
            True,
        ),
        PreparedResidentOperation(
            nodes["wall_loss"],
            _enqueue_prepared_operation_execute,
            (wall_loss,),
            wall_loss,
            True,
        ),
        PreparedResidentOperation(
            nodes["nucleation"],
            _enqueue_prepared_operation_execute,
            (nucleation,),
            nucleation,
            True,
        ),
        PreparedResidentOperation(
            nodes["diagnostics"],
            _enqueue_prepared_diagnostics_window,
            (thermal.diagnostics, diagnostics),
            diagnostics,
            True,
        ),
    )
    result = PreparedResidentSimulation(
        prepared,
        typed,
        prepared.session,
        prepared.registry,
        prepared.guard,
        prepared.lifecycle,
        prepared.signature,
        prepared.graph,
        prepared.schedule,
        _CANONICAL_IDS,
        prepared.primary_arrays,
        prepared.resource_views,
        prepared.capture_requirements,
        prepared.capture_set,
        prepared.capture_report,
        node_sequence,
        thermal,
        communication,
        environment,
        gas,
        condensation,
        coagulation,
        dilution,
        wall_loss,
        nucleation,
        diagnostics,
        operations,
        duration,
    )
    _validate_prepared_resident_simulation(result)
    return result


def _validate_prepared_resident_simulation(prepared: object) -> None:
    """Perform the READY-only identity and signature gate before token entry.

    Args:
        prepared: Candidate prepared simulation carrier whose retained
            identities and setup products must be checked.

    Raises:
        TypeError: If ``prepared`` is not an exact prepared simulation carrier.
        ValueError: If any retained identity, operation, or READY attachment
            metadata has drifted.
    """
    if type(prepared) is not PreparedResidentSimulation:
        raise TypeError("prepared must be an exact PreparedResidentSimulation.")
    typed = cast(PreparedResidentSimulation, prepared)
    if (
        typed.timestep.request is not typed.request
        or typed.request.session is not typed.session
        or typed.request.registry is not typed.registry
        or typed.request.guard is not typed.guard
        or typed.timestep.lifecycle is not typed.lifecycle
        or typed.timestep.signature is not typed.signature
        or typed.request.graph is not typed.graph
        or typed.request.schedule is not typed.schedule
        or tuple(item.node.node_id for item in typed.operations)
        != _CANONICAL_IDS
        or len(typed.nodes) != len(_CANONICAL_IDS)
        or any(
            operation.node is not node
            for operation, node in zip(
                typed.operations, typed.nodes, strict=True
            )
        )
        or typed.ordered_node_ids != _CANONICAL_IDS
    ):
        raise ValueError(
            "prepared resident simulation identities do not match."
        )
    expected_products = (
        typed.communication,
        typed.communication,
        typed.environment,
        typed.gas,
        typed.thermal.condensation,
        typed.thermal.condensation,
        typed.condensation,
        typed.coagulation,
        typed.dilution,
        typed.wall_loss,
        typed.nucleation,
        typed.diagnostics,
    )
    expected_handlers = (
        _enqueue_prepared_resident_communication_node,
        _enqueue_prepared_resident_volume_evolution_node,
        _enqueue_prepared_environment_update,
        _enqueue_prepared_gas_update,
        _enqueue_prepared_vapor_pressure,
        _enqueue_prepared_saturation_ratio,
        _enqueue_prepared_operation_execute,
        _enqueue_prepared_operation_execute,
        _enqueue_prepared_operation_execute,
        _enqueue_prepared_operation_execute,
        _enqueue_prepared_operation_execute,
        _enqueue_prepared_diagnostics_window,
    )
    expected_arguments = (
        (typed.communication,),
        (typed.communication,),
        (typed.environment,),
        (typed.gas,),
        (typed.thermal.condensation,),
        (typed.thermal.condensation,),
        (typed.condensation,),
        (typed.coagulation,),
        (typed.dilution,),
        (typed.wall_loss,),
        (typed.nucleation,),
        (typed.thermal.diagnostics, typed.diagnostics),
    )
    if any(
        operation.product is not product
        or operation.handler is not handler
        or operation.writer_capable is not writer_capable
        or len(operation.arguments) != len(arguments)
        or any(
            argument is not expected
            for argument, expected in zip(
                operation.arguments, arguments, strict=True
            )
        )
        for operation, product, handler, arguments, writer_capable in zip(
            typed.operations,
            expected_products,
            expected_handlers,
            expected_arguments,
            _CANONICAL_WRITER_CAPABLE,
            strict=True,
        )
    ):
        raise ValueError("prepared resident simulation products do not match.")
    if (
        typed.primary_arrays is not typed.timestep.primary_arrays
        or typed.resource_views is not typed.timestep.resource_views
        or typed.capture_requirements
        is not typed.request.capture_resource_requirements
        or typed.capture_requirements is not typed.timestep.capture_requirements
        or typed.capture_set is None
        or typed.capture_set is not typed.timestep.capture_set
        or typed.capture_report is not typed.timestep.capture_report
        or typed.capture_report is None
        or cast(Any, typed.capture_set).report is not typed.capture_report
        or typed.duration != typed.timestep.duration
    ):
        raise ValueError(
            "prepared resident simulation metadata identities do not match."
        )
    _validate_ready_attachment(
        typed.request,
        typed.timestep.binding,
        typed.lifecycle,
        typed.signature,
        typed.session,
        typed.registry,
        typed.guard,
    )
    from particula.execution.graph_capture import (
        validate_resident_capture_resources,
    )

    if (
        validate_resident_capture_resources(typed.request)
        is not typed.capture_set
    ):
        raise ValueError("prepared capture resource set does not match.")
    compatibility = compare_resident_graph_capture_signature(
        cast(Any, typed.signature),
        typed.request,
        admission_token=cast(Any, typed.signature),
    )
    if not compatibility.compatible:
        raise ValueError("resident graph-capture signature is incompatible.")


def enqueue_prepared_resident_simulation(prepared: object) -> None:
    """Enqueue one prepared READY simulation under exactly one token.

    This composed scheduler boundary is distinct from an individual prepared
    binding: its mandatory pre-token gate rechecks retained READY lifecycle,
    attachment, and structural identities before opening the token. After
    admission, it invokes the twelve setup-bound operations in canonical order
    without repeating host/setup metadata validation or performing host
    readback, allocation, transfer, synchronization, lookup, rebinding, or RNG
    reset. Retained native operations still perform device-side status and
    physical-state validation. The scheduler then closes the token exactly once.
    Rejection before a writer-capable invocation preserves the active session;
    a later failure follows the FAULTED no-rollback, no-retry, and no-fallback
    resident recovery boundary.

    Args:
        prepared: Exact prepared simulation carrier returned by
            :func:`prepare_resident_simulation`.

    Raises:
        TypeError: If ``prepared`` is not an exact prepared simulation carrier.
        ValueError: If retained identities or lifecycle metadata have drifted,
            or a fresh signature comparison detects structural drift before
            token entry.
        BaseException: Propagates an operation failure after lifecycle cleanup.
    """
    _validate_prepared_resident_simulation(prepared)
    typed = cast(PreparedResidentSimulation, prepared)
    token = cast(Any, typed.guard).begin_step(typed.duration)
    writer_called = False
    try:
        for operation, writer_capable in zip(
            typed.operations, _CANONICAL_WRITER_CAPABLE, strict=True
        ):
            writer_called = writer_called or writer_capable
            operation.handler(*operation.arguments)
            record_prepared_thermodynamic_success(
                cast(Any, typed.thermal), operation.node
            )
        cast(Any, typed.guard).complete_step(token)
    except BaseException as error:
        _cleanup_resident_execution_failure(
            typed.request, token, writer_called, error, classify_capture=True
        )


def _enqueue_captured_prepared_operations(prepared: object) -> None:
    """Dispatch retained prepared operations inside an active native capture.

    The graph-capture owner validates ``prepared`` before opening its native
    capture window. This helper deliberately performs only the frozen operation
    calls, in their retained canonical order; it does not validate, open a
    guard token, record thermodynamic progress, or clean up failures.

    Args:
        prepared: Already-qualified exact prepared resident simulation.

    Raises:
        BaseException: Propagates an exception raised by a retained operation.
    """
    typed = cast(PreparedResidentSimulation, prepared)
    for operation in typed.operations:
        operation.handler(*operation.arguments)


def _cleanup_resident_execution_failure(
    request: ResidentSimulationRequest,
    token: object,
    writer_called: bool,
    error: BaseException,
    *,
    classify_capture: bool,
) -> None:
    """Close a failed token and preserve legacy lifecycle classification."""
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
            cast(ResidentStepToken, token),
            outcome,
        )
    except BaseException as cleanup_error:
        raise error from cleanup_error
    if (
        classify_capture
        and outcome is _ResidentOperationOutcome.WRITER_MAY_HAVE_LAUNCHED
        and request.graph_capture_binding is not None
    ):
        try:
            classify_resident_graph_capture_writer_failure(
                request.graph_capture_binding
            )
        except BaseException as classification_error:
            _fault_resident_graph_capture_after_classification_failure(
                request.graph_capture_binding
            )
            raise error from classification_error
    raise error


def _enqueue_prepared_diagnostics_window(
    consumer: object,
    diagnostics: object,
) -> None:
    """Enqueue retained diagnostic saturation-refresh and copy operations."""
    _enqueue_prepared_saturation_ratio(cast(Any, consumer))
    _enqueue_prepared_resident_diagnostics(cast(Any, diagnostics))


def _enqueue_prepared_operation_execute(operation: object) -> object:
    """Invoke one setup-bound prepared operation without resolving it again.

    Args:
        operation: Prepared adapter operation retaining its execution product.

    Returns:
        The result returned by the retained operation's ``execute`` method.
    """
    return cast(Any, operation).execute()


class ResidentSimulationScheduler:
    """Execute one canonical fully resolved resident timestep at a time.

    Each successful call opens and completes exactly one lifecycle token while
    dispatching the resolved twelve-node schedule. Communication runs with
    pre-update volumes, optional volume evolution follows it, and both barriers
    invalidate saturation ratio only. Nucleation completes through its ordinary
    adapter; only condensation and diagnostics are thermodynamic consumers.
    The scheduler neither transfers nor
    restores data, acquires resources, synchronizes, retries, falls back, or
    rolls back after a writer-capable operation may have launched. It dispatches
    already-published coagulation and wall-loss RNG sidecars by identity with
    reset disabled. Scheduler-resolved wall-loss selection reaches the adapter,
    which excludes disabled logical-box lanes from direct dispatch. The
    scheduler neither allocates, reseeds, inspects, nor synchronizes either
    stream. An optional attached graph-capture binding is only a pre-dispatch
    metadata gate: the scheduler does not capture or replay graphs, recapture,
    transfer, synchronize, fall back, or replace resources for it.
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

        An attached graph-capture binding is checked after the resident guard
        and registry links, but before graph and process validation or token
        entry. A failed gate therefore cannot dispatch an adapter or mutate
        lifecycle state except for deliberate structural-drift invalidation.

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
        if request.graph_capture_binding is not None:
            binding = cast(Any, request.graph_capture_binding)
            if binding._request is not request:
                raise ValueError(
                    "graph-capture binding must retain the executing request."
                )
            gate_resident_graph_capture(binding)
        _validate_complete_resident_timestep_metadata(request, duration)

    def _validate_request_nodes(
        self, graph_by_id: dict[str, ProcessNode]
    ) -> None:
        """Validate request bindings against exact resolved graph nodes."""
        _validate_resident_request_nodes(self._request, graph_by_id)

    def _validate_durations(self, duration: Real) -> None:
        """Require every process request to retain the exact step duration."""
        _validate_resident_durations(self._request, duration)

    @staticmethod
    def _validate_virtual_refresh_windows(
        ids: tuple[str, ...], dependencies: tuple[DependencyEdge, ...]
    ) -> None:
        """Require the complete resolver freshness window."""
        _validate_virtual_refresh_windows(ids, dependencies)

    def execute(self, duration: object) -> None:  # noqa: C901
        """Preflight then run one complete ordered timestep.

        Failures before a writer-capable invocation leave the session active.
        Once dispatch begins, the token is closed and the session faults without
        rollback because a native writer may already have launched. For an
        attached binding, the existing writer-failure classification is recorded
        only after cleanup confirms that outcome; no capture, replay, retry, or
        fallback occurs.

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
        coagulation = ResidentBrownianCoagulationExecutionAdapter()
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
                    coagulation.execute(cast(Any, request.coagulation))
                    thermal.record_completed(node)
                elif node_id == "dilution":
                    dilution.execute(request.dilution)
                    thermal.record_completed(node)
                elif node_id == "wall_loss":
                    wall_loss.execute(request.wall_loss)
                    thermal.record_completed(node)
                elif node_id == "nucleation":
                    nucleation.execute(request.nucleation)
                    thermal.record_completed(node)
                elif node_id == "diagnostics":
                    thermal.execute_consumer(
                        node, lambda: diagnostics.execute(request.diagnostics)
                    )
            writer_called = True
            request.guard.complete_step(token)
        except BaseException as error:
            _cleanup_resident_execution_failure(
                request, token, writer_called, error, classify_capture=True
            )


def _validate_complete_resident_timestep_metadata(
    request: ResidentSimulationRequest, duration: Real
) -> None:
    """Validate complete-loop metadata without scheduler construction.

    This shared read-only seam validates the resolver-produced canonical
    schedule, retained request bindings, diagnostics, communication, and
    process-duration agreement. It does not create executors, enter a guard,
    acquire resources, inspect payloads, dispatch work, or mutate lifecycle
    state.

    Args:
        request: Exact resident request with an attached complete-loop graph.
        duration: Already validated finite, nonnegative timestep duration.

    Raises:
        ValueError: If graph, schedule, request bindings, metadata, or duration
            agreement is invalid.
    """
    from particula.execution.graph_capture import (
        validate_resident_capture_resources,
    )

    validate_resident_capture_resources(request)
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
        if graph_by_id.get(node.node_id) is not node:
            raise ValueError("schedule nodes must be identical graph members.")
    _validate_virtual_refresh_windows(ids, request.schedule.dependencies)
    _validate_resident_request_nodes(request, graph_by_id)
    validate_resident_diagnostics_plan(request.diagnostics)
    _validate_resident_durations(request, duration)


def _validate_virtual_refresh_windows(
    ids: tuple[str, ...], dependencies: tuple[DependencyEdge, ...]
) -> None:
    """Require the resolver's canonical thermodynamic refresh windows.

    Args:
        ids: Canonically ordered resolved node identifiers.
        dependencies: Resolved directed dependency edges.

    Raises:
        ValueError: If virtual refresh nodes do not retain their required order
            and dependency edges.
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


def _validate_resident_request_nodes(  # noqa: C901
    request: ResidentSimulationRequest, graph_by_id: dict[str, ProcessNode]
) -> None:
    """Validate request bindings against resolved graph and resources.

    Args:
        request: Exact resident request whose retained bindings are checked.
        graph_by_id: Exact resolved graph nodes keyed by node identifier.

    Raises:
        ValueError: If a process, diagnostic, communication, or published
            resource binding does not match the resident request.
    """
    registry = cast(Any, request.registry)
    for item, node_id in (
        (request.environment_update, "environment_update"),
        (request.gas_update, "gas_update"),
    ):
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
        raise ValueError("diagnostics plan does not match resolved binding.")
    for process_request in (
        request.dilution,
        request.wall_loss,
        request.nucleation,
    ):
        if (
            process_request.session is not request.session
            or process_request.registry is not request.registry
        ):
            raise ValueError("process request does not match resident binding.")
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
        validate_resident_communication_request(communication)
    condensation = request.condensation.state
    if (
        condensation.particles is not request.session.particles
        or condensation.gas is not request.session.gas
        or condensation.environment is not request.session.environment
        or condensation.thermodynamics is not request.thermodynamics
    ):
        raise ValueError("condensation state does not match resident binding.")
    coagulation_request = request.coagulation
    coagulation = coagulation_request.request.state
    if (
        coagulation.particles is not request.session.particles
        or coagulation.environment is not request.session.environment
    ):
        raise ValueError("coagulation state does not match resident binding.")
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
    if (
        coagulation_request.session is not request.session
        or coagulation_request.registry is not request.registry
        or coagulation_request.resources is not coagulation_resources
    ):
        raise ValueError("coagulation request does not match resident binding.")
    registry.validate_wall_loss_resources(
        request.session, request.wall_loss.resources
    )
    request.wall_loss.validate_enabled_box_indices()
    registry.validate_nucleation_resources(
        request.session, request.nucleation.resources
    )


def _validate_resident_durations(
    request: ResidentSimulationRequest, duration: Real
) -> None:
    """Require every process request to retain the exact step duration.

    Args:
        request: Exact resident request containing process timestep values.
        duration: Already validated duration required by every process.

    Raises:
        ValueError: If any process timestep differs from ``duration``.
    """
    values = (
        request.condensation.time_step,
        request.coagulation.request.state.time_step,
        request.dilution.time_step,
        request.wall_loss.time_step,
        request.nucleation.time_step,
    )
    if any(value != duration for value in values):
        raise ValueError("all process time_step values must equal duration.")
