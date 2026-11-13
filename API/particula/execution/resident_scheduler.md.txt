# particula.execution.resident_scheduler

*module*

Run legacy and READY-prepared twelve-node GPU-resident schedules.

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

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L1-L1)

## particula.execution.resident_scheduler._CANONICAL_IDS

*attribute* · *module attribute*

```python
_CANONICAL_IDS = ('communication', 'volume_evolution', 'environment_update', 'gas_update', 'vapor_pressure_refresh', 'saturation_refresh', 'condensation', 'brownian_coagulation', 'dilution', 'wall_loss', 'nucleation', 'diagnostics')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L131-L144)

## particula.execution.resident_scheduler._CANONICAL_WRITER_CAPABLE

*attribute* · *module attribute*

```python
_CANONICAL_WRITER_CAPABLE = (True) * len(_CANONICAL_IDS)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L145-L145)

## particula.execution.resident_scheduler._COMPLETE_IDS

*attribute* · *module attribute*

```python
_COMPLETE_IDS = frozenset({'communication', 'volume_evolution', 'environment_update', 'gas_update', 'vapor_pressure_refresh', 'saturation_refresh', 'condensation', 'brownian_coagulation', 'dilution', 'wall_loss', 'nucleation', 'diagnostics'})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L114-L129)

## particula.execution.resident_scheduler._VIRTUAL_IDS

*attribute* · *module attribute*

```python
_VIRTUAL_IDS = frozenset({'vapor_pressure_refresh', 'saturation_refresh'})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L130-L130)

## particula.execution.resident_scheduler.PreparedResidentOperation

*class* · *dataclass*

```python
class PreparedResidentOperation
```

Retain one setup-validated resident operation for direct dispatch.

**Attributes**

- `node` (`ProcessNode`) — Resolver-produced node represented by this operation.
- `handler` (`Callable[..., object]`) — Scheduler-owned fixed handler for the prepared operation.
- `arguments` (`tuple[object, ...]`) — Setup-bound arguments supplied to ``handler`` at dispatch.
- `product` (`object`) — Setup product retained by identity for drift detection.
- `writer_capable` (`bool`) — Whether invoking ``handler`` may launch a device writer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L148-L164)

### particula.execution.resident_scheduler.PreparedResidentOperation.arguments

*attribute* · *instance attribute*

```python
arguments: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L162-L162)

### particula.execution.resident_scheduler.PreparedResidentOperation.handler

*attribute* · *instance attribute*

```python
handler: Callable[..., object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L161-L161)

### particula.execution.resident_scheduler.PreparedResidentOperation.node

*attribute* · *instance attribute*

```python
node: ProcessNode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L160-L160)

### particula.execution.resident_scheduler.PreparedResidentOperation.product

*attribute* · *instance attribute*

```python
product: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L163-L163)

### particula.execution.resident_scheduler.PreparedResidentOperation.writer_capable

*attribute* · *instance attribute*

```python
writer_capable: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L164-L164)

### particula.execution.resident_scheduler.PreparedResidentOperation.__init__

*method*

```python
def __init__(node: ProcessNode, handler: Callable[..., object], arguments: tuple[object, ...], product: object, writer_capable: bool) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L0-L0)

## particula.execution.resident_scheduler.PreparedResidentSimulation

*class* · *dataclass*

```python
class PreparedResidentSimulation
```

Freeze all READY-bound products required for one uncaptured timestep.

The carrier is identity-semantic: its metadata, setup products, node order,
and operation callables are all retained during preparation. Enqueue can
therefore dispatch the complete schedule without reconstructing executors,
resolving nodes, repeating payload validation, or invoking graph-capture
admission. It represents a READY attachment, not a captured graph.

**Attributes**

- `timestep` (`PreparedResidentTimestep`) — Prepared declaration timestep retained by identity.
- `request` (`ResidentSimulationRequest`) — Complete resident request retained by identity.
- `session` (`object`) — Resident session owning the dispatched containers.
- `registry` (`GPUResourceRegistry`) — Resource registry pinned to ``session``.
- `guard` (`object`) — Closed step guard for the exact session and registry binding.
- `lifecycle` (`object`) — READY lifecycle metadata retained from the declaration timestep.
- `signature` (`object`) — Prepared graph-capture signature metadata.
- `graph` (`object`) — Resolver-produced graph for the complete loop.
- `schedule` (`object`) — Resolver-produced schedule for the complete loop.
- `ordered_node_ids` (`tuple[object, ...]`) — Canonical twelve-node operation identifiers.
- `primary_arrays` (`tuple[object, ...]`) — Resident primary arrays retained by identity.
- `resource_views` (`tuple[object, ...]`) — Published resource views retained by identity.
- `capture_requirements` (`object`) — Pre-published capture requirements retained by identity.
- `capture_set` (`object`) — Registry-published capture resource set retained by identity.
- `capture_report` (`object`) — Cached immutable logical-byte report retained by identity.
- `nodes` (`tuple[ProcessNode, ...]`) — Canonical graph nodes in operation order.
- `thermal` (`PreparedResidentThermodynamicSequence`) — Prepared thermodynamic refresh sequence.
- `communication` (`PreparedResidentCommunicationBinding`) — Prepared communication and volume barriers.
- `environment` (`object`) — Prepared environment update operation.
- `gas` (`object`) — Prepared gas update operation.
- `condensation` (`object`) — Prepared condensation operation.
- `coagulation` (`object`) — Prepared Brownian coagulation operation.
- `dilution` (`object`) — Prepared dilution operation.
- `wall_loss` (`object`) — Prepared wall-loss operation.
- `nucleation` (`object`) — Prepared nucleation operation.
- `diagnostics` (`object`) — Prepared diagnostics operation.
- `operations` (`tuple[PreparedResidentOperation, ...]`) — Ordered callables dispatched during enqueue.
- `duration` (`object`) — Nonnegative timestep duration in seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L167-L239)

### particula.execution.resident_scheduler.PreparedResidentSimulation.capture_report

*attribute* · *instance attribute*

```python
capture_report: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L226-L226)

### particula.execution.resident_scheduler.PreparedResidentSimulation.capture_requirements

*attribute* · *instance attribute*

```python
capture_requirements: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L224-L224)

### particula.execution.resident_scheduler.PreparedResidentSimulation.capture_set

*attribute* · *instance attribute*

```python
capture_set: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L225-L225)

### particula.execution.resident_scheduler.PreparedResidentSimulation.coagulation

*attribute* · *instance attribute*

```python
coagulation: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L233-L233)

### particula.execution.resident_scheduler.PreparedResidentSimulation.communication

*attribute* · *instance attribute*

```python
communication: PreparedResidentCommunicationBinding
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L229-L229)

### particula.execution.resident_scheduler.PreparedResidentSimulation.condensation

*attribute* · *instance attribute*

```python
condensation: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L232-L232)

### particula.execution.resident_scheduler.PreparedResidentSimulation.diagnostics

*attribute* · *instance attribute*

```python
diagnostics: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L237-L237)

### particula.execution.resident_scheduler.PreparedResidentSimulation.dilution

*attribute* · *instance attribute*

```python
dilution: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L234-L234)

### particula.execution.resident_scheduler.PreparedResidentSimulation.duration

*attribute* · *instance attribute*

```python
duration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L239-L239)

### particula.execution.resident_scheduler.PreparedResidentSimulation.environment

*attribute* · *instance attribute*

```python
environment: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L230-L230)

### particula.execution.resident_scheduler.PreparedResidentSimulation.gas

*attribute* · *instance attribute*

```python
gas: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L231-L231)

### particula.execution.resident_scheduler.PreparedResidentSimulation.graph

*attribute* · *instance attribute*

```python
graph: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L219-L219)

### particula.execution.resident_scheduler.PreparedResidentSimulation.guard

*attribute* · *instance attribute*

```python
guard: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L216-L216)

### particula.execution.resident_scheduler.PreparedResidentSimulation.lifecycle

*attribute* · *instance attribute*

```python
lifecycle: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L217-L217)

### particula.execution.resident_scheduler.PreparedResidentSimulation.nodes

*attribute* · *instance attribute*

```python
nodes: tuple[ProcessNode, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L227-L227)

### particula.execution.resident_scheduler.PreparedResidentSimulation.nucleation

*attribute* · *instance attribute*

```python
nucleation: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L236-L236)

### particula.execution.resident_scheduler.PreparedResidentSimulation.operations

*attribute* · *instance attribute*

```python
operations: tuple[PreparedResidentOperation, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L238-L238)

### particula.execution.resident_scheduler.PreparedResidentSimulation.ordered_node_ids

*attribute* · *instance attribute*

```python
ordered_node_ids: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L221-L221)

### particula.execution.resident_scheduler.PreparedResidentSimulation.primary_arrays

*attribute* · *instance attribute*

```python
primary_arrays: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L222-L222)

### particula.execution.resident_scheduler.PreparedResidentSimulation.registry

*attribute* · *instance attribute*

```python
registry: GPUResourceRegistry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L215-L215)

### particula.execution.resident_scheduler.PreparedResidentSimulation.request

*attribute* · *instance attribute*

```python
request: ResidentSimulationRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L213-L213)

### particula.execution.resident_scheduler.PreparedResidentSimulation.resource_views

*attribute* · *instance attribute*

```python
resource_views: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L223-L223)

### particula.execution.resident_scheduler.PreparedResidentSimulation.schedule

*attribute* · *instance attribute*

```python
schedule: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L220-L220)

### particula.execution.resident_scheduler.PreparedResidentSimulation.session

*attribute* · *instance attribute*

```python
session: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L214-L214)

### particula.execution.resident_scheduler.PreparedResidentSimulation.signature

*attribute* · *instance attribute*

```python
signature: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L218-L218)

### particula.execution.resident_scheduler.PreparedResidentSimulation.thermal

*attribute* · *instance attribute*

```python
thermal: PreparedResidentThermodynamicSequence
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L228-L228)

### particula.execution.resident_scheduler.PreparedResidentSimulation.timestep

*attribute* · *instance attribute*

```python
timestep: PreparedResidentTimestep
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L212-L212)

### particula.execution.resident_scheduler.PreparedResidentSimulation.wall_loss

*attribute* · *instance attribute*

```python
wall_loss: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L235-L235)

### particula.execution.resident_scheduler.PreparedResidentSimulation.__init__

*method*

```python
def __init__(timestep: PreparedResidentTimestep, request: ResidentSimulationRequest, session: object, registry: GPUResourceRegistry, guard: object, lifecycle: object, signature: object, graph: object, schedule: object, ordered_node_ids: tuple[object, ...], primary_arrays: tuple[object, ...], resource_views: tuple[object, ...], capture_requirements: object, capture_set: object, capture_report: object, nodes: tuple[ProcessNode, ...], thermal: PreparedResidentThermodynamicSequence, communication: PreparedResidentCommunicationBinding, environment: object, gas: object, condensation: object, coagulation: object, dilution: object, wall_loss: object, nucleation: object, diagnostics: object, operations: tuple[PreparedResidentOperation, ...], duration: object) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L0-L0)

## particula.execution.resident_scheduler.ResidentSimulationRequest

*class* · *dataclass*

```python
class ResidentSimulationRequest
```

Bind one complete resolved simulation loop to resident resources.

The request retains all carriers by identity for the canonical twelve-node
graph. It is concrete-only and does not acquire resources, begin a step,
transfer, synchronize, or validate physical process inputs.

**Attributes**

- `session` (`ResidentSession`) — Exact active resident session.
- `registry` (`GPUResourceRegistry`) — Exact registry pinned to ``session``.
- `guard` (`ResidentStepGuard`) — Exact closed lifecycle guard for the same binding.
- `graph` (`ResolvedProcessGraph`) — Resolver-produced process graph for the complete loop.
- `schedule` (`ResolvedTimestepSchedule`) — Canonical resolved schedule for ``graph``.
- `thermodynamics` (`ThermodynamicsConfig`) — Exact configuration shared by thermal consumers.
- `condensation` (`WarpCondensationExecutionState`) — Exact resident condensation execution state.
- `coagulation` (`ResidentBrownianCoagulationExecutionState`) — Resident Brownian execution state binding the published coagulation-only RNG sidecar by identity and requiring forced-false RNG initialization during dispatch.
- `dilution` (`ResidentDilutionRequest`) — Exact resident dilution request.
- `wall_loss` (`ResidentWallLossRequest`) — Exact resident wall-loss request retaining the published wall-loss RNG sidecar by identity. Its reset flag must be literal ``False`` and its scheduler-owned selected logical-box indices must be validated in ascending order before dispatch.
- `nucleation` (`ResidentNucleationRequest`) — Exact resident nucleation request.
- `diagnostics` (`ResidentDiagnosticsPlan`) — Exact closed diagnostics plan.
- `environment_update` (`ResidentEnvironmentUpdateRequest | None`) — Optional exact environment update request.
- `gas_update` (`ResidentGasUpdateRequest | None`) — Optional exact gas update request.
- `communication` (`ResidentCommunicationRequest | None`) — Exact request for the communication and volume barriers.
- `capture_resource_requirements` (`CaptureResourceRequirements`) — Exact pre-published capture resource requirements retained for metadata-only admission checks.
- `graph_capture_binding` (`object | None`) — Optional exact direct-module-only binding. It is attached only after final request construction and gates admission; it neither captures nor replays graphs nor changes dispatch.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L267-L381)

### particula.execution.resident_scheduler.ResidentSimulationRequest.capture_resource_requirements

*attribute* · *instance attribute*

```python
capture_resource_requirements: CaptureResourceRequirements
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L318-L318)

### particula.execution.resident_scheduler.ResidentSimulationRequest.coagulation

*attribute* · *instance attribute*

```python
coagulation: ResidentBrownianCoagulationExecutionState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L310-L310)

### particula.execution.resident_scheduler.ResidentSimulationRequest.communication

*attribute* · *instance attribute*

```python
communication: ResidentCommunicationRequest | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L317-L317)

### particula.execution.resident_scheduler.ResidentSimulationRequest.condensation

*attribute* · *instance attribute*

```python
condensation: WarpCondensationExecutionState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L309-L309)

### particula.execution.resident_scheduler.ResidentSimulationRequest.diagnostics

*attribute* · *instance attribute*

```python
diagnostics: ResidentDiagnosticsPlan
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L314-L314)

### particula.execution.resident_scheduler.ResidentSimulationRequest.dilution

*attribute* · *instance attribute*

```python
dilution: ResidentDilutionRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L311-L311)

### particula.execution.resident_scheduler.ResidentSimulationRequest.environment_update

*attribute* · *instance attribute*

```python
environment_update: ResidentEnvironmentUpdateRequest | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L315-L315)

### particula.execution.resident_scheduler.ResidentSimulationRequest.gas_update

*attribute* · *instance attribute*

```python
gas_update: ResidentGasUpdateRequest | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L316-L316)

### particula.execution.resident_scheduler.ResidentSimulationRequest.graph

*attribute* · *instance attribute*

```python
graph: ResolvedProcessGraph
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L306-L306)

### particula.execution.resident_scheduler.ResidentSimulationRequest.graph_capture_binding

*attribute* · *class attribute* · *instance attribute*

```python
graph_capture_binding: object | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L319-L319)

### particula.execution.resident_scheduler.ResidentSimulationRequest.guard

*attribute* · *instance attribute*

```python
guard: ResidentStepGuard
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L305-L305)

### particula.execution.resident_scheduler.ResidentSimulationRequest.nucleation

*attribute* · *instance attribute*

```python
nucleation: ResidentNucleationRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L313-L313)

### particula.execution.resident_scheduler.ResidentSimulationRequest.registry

*attribute* · *instance attribute*

```python
registry: GPUResourceRegistry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L304-L304)

### particula.execution.resident_scheduler.ResidentSimulationRequest.schedule

*attribute* · *instance attribute*

```python
schedule: ResolvedTimestepSchedule
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L307-L307)

### particula.execution.resident_scheduler.ResidentSimulationRequest.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L303-L303)

### particula.execution.resident_scheduler.ResidentSimulationRequest.thermodynamics

*attribute* · *instance attribute*

```python
thermodynamics: ThermodynamicsConfig
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L308-L308)

### particula.execution.resident_scheduler.ResidentSimulationRequest.wall_loss

*attribute* · *instance attribute*

```python
wall_loss: ResidentWallLossRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L312-L312)

### particula.execution.resident_scheduler.ResidentSimulationRequest.__init__

*method*

```python
def __init__(session: ResidentSession, registry: GPUResourceRegistry, guard: ResidentStepGuard, graph: ResolvedProcessGraph, schedule: ResolvedTimestepSchedule, thermodynamics: ThermodynamicsConfig, condensation: WarpCondensationExecutionState, coagulation: ResidentBrownianCoagulationExecutionState, dilution: ResidentDilutionRequest, wall_loss: ResidentWallLossRequest, nucleation: ResidentNucleationRequest, diagnostics: ResidentDiagnosticsPlan, environment_update: ResidentEnvironmentUpdateRequest | None, gas_update: ResidentGasUpdateRequest | None, communication: ResidentCommunicationRequest | None, capture_resource_requirements: CaptureResourceRequirements, graph_capture_binding: object | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L0-L0)

### particula.execution.resident_scheduler.ResidentSimulationRequest.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact request components and optional update types.

**Raises**

- `TypeError` — If a required component or optional update has an
inexact concrete type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L321-L381)

## particula.execution.resident_scheduler.ResidentSimulationScheduler

*class*

```python
class ResidentSimulationScheduler
```

Execute one canonical fully resolved resident timestep at a time.

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

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L832-L1018)

### particula.execution.resident_scheduler.ResidentSimulationScheduler._request

*attribute* · *instance attribute*

```python
_request = request
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L865-L865)

### particula.execution.resident_scheduler.ResidentSimulationScheduler.__init__

*method*

```python
def __init__(request: ResidentSimulationRequest) -> None
```

Retain one exact resident simulation request.

**Parameters**

- `request` (`ResidentSimulationRequest`) — Complete identity-bound resident simulation request.

**Raises**

- `TypeError` — If ``request`` is not an exact request instance.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L852-L865)

### particula.execution.resident_scheduler.ResidentSimulationScheduler._validate

*method*

```python
def _validate(duration: object) -> None
```

Preflight the lifecycle, graph, request, and duration bindings.

An attached graph-capture binding is checked after the resident guard
and registry links, but before graph and process validation or token
entry. A failed gate therefore cannot dispatch an adapter or mutate
lifecycle state except for deliberate structural-drift invalidation.

**Parameters**

- `duration` (`object`) — Candidate nonnegative finite timestep duration.

**Raises**

- `TypeError` — If the duration is not a non-boolean real value.
- `ValueError` — If duration, ownership, graph, schedule, request, or
diagnostics validation fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L867-L906)

### particula.execution.resident_scheduler.ResidentSimulationScheduler._validate_durations

*method*

```python
def _validate_durations(duration: Real) -> None
```

Require every process request to retain the exact step duration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L914-L916)

### particula.execution.resident_scheduler.ResidentSimulationScheduler._validate_request_nodes

*method*

```python
def _validate_request_nodes(graph_by_id: dict[str, ProcessNode]) -> None
```

Validate request bindings against exact resolved graph nodes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L908-L912)

### particula.execution.resident_scheduler.ResidentSimulationScheduler._validate_virtual_refresh_windows

*method* · *staticmethod*

```python
def _validate_virtual_refresh_windows(ids: tuple[str, ...], dependencies: tuple[DependencyEdge, ...]) -> None
```

Require the complete resolver freshness window.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L918-L923)

### particula.execution.resident_scheduler.ResidentSimulationScheduler.execute

*method*

```python
def execute(duration: object) -> None
```

Preflight then run one complete ordered timestep.

Failures before a writer-capable invocation leave the session active.
Once dispatch begins, the token is closed and the session faults without
rollback because a native writer may already have launched. For an
attached binding, the existing writer-failure classification is recorded
only after cleanup confirms that outcome; no capture, replay, retry, or
fallback occurs.

**Parameters**

- `duration` (`object`) — Nonnegative finite duration matching each process request.

**Raises**

- `TypeError` — If preflight finds an inexact carrier or invalid
duration.
- `ValueError` — If resolved bindings, duration agreement, or an invoked
operation reject execution.
- `RuntimeError` — If lifecycle token handling rejects the timestep.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L925-L1018)

## particula.execution.resident_scheduler._capture_resource_requirements_type

*function*

```python
def _capture_resource_requirements_type() -> type[object]
```

Return the concrete capture requirements type without an import cycle.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L260-L264)

## particula.execution.resident_scheduler._cleanup_resident_execution_failure

*function*

```python
def _cleanup_resident_execution_failure(request: ResidentSimulationRequest, token: object, writer_called: bool, error: BaseException, *, classify_capture: bool) -> None
```

Close a failed token and preserve legacy lifecycle classification.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L770-L808)

## particula.execution.resident_scheduler._enqueue_captured_prepared_operations

*function*

```python
def _enqueue_captured_prepared_operations(prepared: object) -> None
```

Dispatch retained prepared operations inside an active native capture.

The graph-capture owner validates ``prepared`` before opening its native
capture window. This helper deliberately performs only the frozen operation
calls, in their retained canonical order; it does not validate, open a
guard token, record thermodynamic progress, or clean up failures.

**Parameters**

- `prepared` (`object`) — Already-qualified exact prepared resident simulation.

**Raises**

- `BaseException` — Propagates an exception raised by a retained operation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L751-L767)

## particula.execution.resident_scheduler._enqueue_prepared_diagnostics_window

*function*

```python
def _enqueue_prepared_diagnostics_window(consumer: object, diagnostics: object) -> None
```

Enqueue retained diagnostic saturation-refresh and copy operations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L811-L817)

## particula.execution.resident_scheduler._enqueue_prepared_operation_execute

*function*

```python
def _enqueue_prepared_operation_execute(operation: object) -> object
```

Invoke one setup-bound prepared operation without resolving it again.

**Parameters**

- `operation` (`object`) — Prepared adapter operation retaining its execution product.

**Returns**

- (`object`) — The result returned by the retained operation's ``execute`` method.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L820-L829)

## particula.execution.resident_scheduler._graph_capture_binding_type

*function*

```python
def _graph_capture_binding_type() -> type[object]
```

Return the concrete optional graph-capture binding type lazily.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L253-L257)

## particula.execution.resident_scheduler._registry_type

*function*

```python
def _registry_type() -> type[object]
```

Return the concrete registry type without creating an import cycle.

**Returns**

- (`type[object]`) — The direct-module-only GPU resource registry type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L242-L250)

## particula.execution.resident_scheduler._validate_complete_resident_timestep_metadata

*function*

```python
def _validate_complete_resident_timestep_metadata(request: ResidentSimulationRequest, duration: Real) -> None
```

Validate complete-loop metadata without scheduler construction.

This shared read-only seam validates the resolver-produced canonical
schedule, retained request bindings, diagnostics, communication, and
process-duration agreement. It does not create executors, enter a guard,
acquire resources, inspect payloads, dispatch work, or mutate lifecycle
state.

**Parameters**

- `request` (`ResidentSimulationRequest`) — Exact resident request with an attached complete-loop graph.
- `duration` (`Real`) — Already validated finite, nonnegative timestep duration.

**Raises**

- `ValueError` — If graph, schedule, request bindings, metadata, or duration
agreement is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L1021-L1072)

## particula.execution.resident_scheduler._validate_prepared_resident_simulation

*function*

```python
def _validate_prepared_resident_simulation(prepared: object) -> None
```

Perform the READY-only identity and signature gate before token entry.

**Parameters**

- `prepared` (`object`) — Candidate prepared simulation carrier whose retained identities and setup products must be checked.

**Raises**

- `TypeError` — If ``prepared`` is not an exact prepared simulation carrier.
- `ValueError` — If any retained identity, operation, or READY attachment
metadata has drifted.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L561-L702)

## particula.execution.resident_scheduler._validate_resident_durations

*function*

```python
def _validate_resident_durations(request: ResidentSimulationRequest, duration: Real) -> None
```

Require every process request to retain the exact step duration.

**Parameters**

- `request` (`ResidentSimulationRequest`) — Exact resident request containing process timestep values.
- `duration` (`Real`) — Already validated duration required by every process.

**Raises**

- `ValueError` — If any process timestep differs from ``duration``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L1227-L1247)

## particula.execution.resident_scheduler._validate_resident_request_nodes

*function*

```python
def _validate_resident_request_nodes(request: ResidentSimulationRequest, graph_by_id: dict[str, ProcessNode]) -> None
```

Validate request bindings against resolved graph and resources.

**Parameters**

- `request` (`ResidentSimulationRequest`) — Exact resident request whose retained bindings are checked.
- `graph_by_id` (`dict[str, ProcessNode]`) — Exact resolved graph nodes keyed by node identifier.

**Raises**

- `ValueError` — If a process, diagnostic, communication, or published
resource binding does not match the resident request.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L1107-L1224)

## particula.execution.resident_scheduler._validate_virtual_refresh_windows

*function*

```python
def _validate_virtual_refresh_windows(ids: tuple[str, ...], dependencies: tuple[DependencyEdge, ...]) -> None
```

Require the resolver's canonical thermodynamic refresh windows.

**Parameters**

- `ids` (`tuple[str, ...]`) — Canonically ordered resolved node identifiers.
- `dependencies` (`tuple[DependencyEdge, ...]`) — Resolved directed dependency edges.

**Raises**

- `ValueError` — If virtual refresh nodes do not retain their required order
and dependency edges.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L1075-L1104)

## particula.execution.resident_scheduler.enqueue_prepared_resident_simulation

*function*

```python
def enqueue_prepared_resident_simulation(prepared: object) -> None
```

Enqueue one prepared READY simulation under exactly one token.

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

**Parameters**

- `prepared` (`object`) — Exact prepared simulation carrier returned by :func:`prepare_resident_simulation`.

**Raises**

- `TypeError` — If ``prepared`` is not an exact prepared simulation carrier.
- `ValueError` — If retained identities or lifecycle metadata have drifted,
or a fresh signature comparison detects structural drift before
token entry.
- `BaseException` — Propagates an operation failure after lifecycle cleanup.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L705-L748)

## particula.execution.resident_scheduler.prepare_resident_simulation

*function*

```python
def prepare_resident_simulation(request: object, duration: object) -> PreparedResidentSimulation
```

Prepare all twelve resident operations while the attachment is READY.

Setup deliberately performs validation and private kernel preparation before
the returned carrier exists. The later enqueue path only invokes the
retained callables under one lifecycle token.

**Parameters**

- `request` (`object`) — Exact complete resident simulation request to prepare.
- `duration` (`object`) — Finite, nonnegative timestep duration in seconds.

**Returns**

- (`PreparedResidentSimulation`) — Frozen identity-semantic carrier containing the twelve prepared
- (`PreparedResidentSimulation`) — operations.

**Raises**

- `TypeError` — If a request or prepared component has an invalid type.
- `ValueError` — If resident metadata, schedule, or resource identities do
not satisfy the READY preparation contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_scheduler.py#L384-L558)
