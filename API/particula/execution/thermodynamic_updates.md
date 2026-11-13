# particula.execution.thermodynamic_updates

*module*

Coordinate resident thermodynamic derived-state freshness.

This direct-import-only Warp boundary consumes resolver-produced graph and
schedule metadata. Legacy callers explicitly report successful ordinary nodes
and bracket supported consumer callbacks. The READY-prepared path validates
and freezes both consumer windows during setup, then enqueues only retained
writers and records successful-node freshness metadata. Both paths refresh
stale vapor-pressure and saturation-ratio fields immediately before a consumer.
Communication and volume-evolution barriers invalidate saturation ratio
without invalidating or refreshing vapor pressure.

It does not own lifecycle, resource acquisition, scheduling, transfers,
fallbacks, or general process dispatch. Prepared setup freezes the consumer
windows and enqueue repeats no host/setup metadata validation or performs
allocation, host readback, transfer, synchronization, mutable-carrier lookup,
or rebinding. Retained refresh writers still perform device-side status and
physical-state validation. Setup rejection is pre-writer; rollback is not
promised after a refresh writer launches. Refreshing vapor pressure delegates to
the authoritative GPU primitive; saturation refresh is a private device writer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L1-L1)

## particula.execution.thermodynamic_updates._CONSUMER_IDS

*attribute* · *module attribute*

```python
_CONSUMER_IDS = frozenset({'condensation', 'diagnostics'})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L521-L521)

## particula.execution.thermodynamic_updates._GAS_CONSTANT

*attribute* · *module attribute*

```python
_GAS_CONSTANT = wp.constant(wp.float64(GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L51-L51)

## particula.execution.thermodynamic_updates._ROLE_SCHEMAS

*attribute* · *module attribute*

```python
_ROLE_SCHEMAS = {'communication': (NodeKind.COMMUNICATION, frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS, ResourceRequirement.PROCESS_SIDECARS}), frozenset({InvalidatedState.SATURATION_RATIO})), 'volume_evolution': (NodeKind.VOLUME_EVOLUTION, frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS, ResourceRequirement.PROCESS_SIDECARS}), frozenset({InvalidatedState.SATURATION_RATIO})), 'environment_update': (NodeKind.ENVIRONMENT_UPDATE, frozenset({ResourceRequirement.ENVIRONMENT}), frozenset({InvalidatedState.VAPOR_PRESSURE, InvalidatedState.SATURATION_RATIO})), 'gas_update': (NodeKind.GAS_UPDATE, frozenset({ResourceRequirement.GAS}), frozenset({InvalidatedState.SATURATION_RATIO})), 'vapor_pressure_refresh': (NodeKind.VAPOR_PRESSURE_REFRESH, frozenset({ResourceRequirement.GAS, ResourceRequirement.ENVIRONMENT, ResourceRequirement.THERMODYNAMICS}), frozenset()), 'saturation_refresh': (NodeKind.SATURATION_REFRESH, frozenset({ResourceRequirement.GAS, ResourceRequirement.ENVIRONMENT, ResourceRequirement.THERMODYNAMICS}), frozenset()), 'condensation': (NodeKind.PROCESS, frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS, ResourceRequirement.ENVIRONMENT, ResourceRequirement.THERMODYNAMICS, ResourceRequirement.PROCESS_SIDECARS}), frozenset({InvalidatedState.SATURATION_RATIO})), 'diagnostics': (NodeKind.DIAGNOSTIC, frozenset({ResourceRequirement.PARTICLES, ResourceRequirement.GAS, ResourceRequirement.ENVIRONMENT, ResourceRequirement.THERMODYNAMICS, ResourceRequirement.DIAGNOSTICS}), frozenset())}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L436-L519)

## particula.execution.thermodynamic_updates._VIRTUAL_IDS

*attribute* · *module attribute*

```python
_VIRTUAL_IDS = frozenset({'vapor_pressure_refresh', 'saturation_refresh'})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L520-L520)

## particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding

*class* · *dataclass*

```python
class PreparedResidentThermodynamicBinding
```

Bind validated declaration thermodynamic writer identities.

This concrete-only record retains the exact arrays and configuration
selected during setup. Enqueue uses those references only and does not
rediscover resident containers or resolve another refresh policy.

**Attributes**

- `request` (`ResidentThermodynamicUpdateRequest`) — Exact thermodynamic update request validated during setup.
- `gas` (`object`) — Resident gas container retained for vapor-pressure refresh.
- `gas_concentration` (`object`) — Bound resident gas concentration array.
- `gas_molar_mass` (`object`) — Bound resident gas molar-mass array.
- `gas_vapor_pressure` (`object`) — Bound resident vapor-pressure destination array.
- `environment_temperature` (`object`) — Bound resident temperature array in K.
- `environment_saturation_ratio` (`object`) — Bound saturation-ratio destination array.
- `device` (`object`) — Device shared by all bound writer arrays.
- `dimensions` (`object`) — Exact resident dimensions used for empty-schema checks.
- `thermodynamics` (`ThermodynamicsConfig`) — Exact configuration for vapor-pressure refresh.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L136-L166)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.device

*attribute* · *instance attribute*

```python
device: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L164-L164)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.dimensions

*attribute* · *instance attribute*

```python
dimensions: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L165-L165)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.environment_saturation_ratio

*attribute* · *instance attribute*

```python
environment_saturation_ratio: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L163-L163)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.environment_temperature

*attribute* · *instance attribute*

```python
environment_temperature: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L162-L162)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.gas

*attribute* · *instance attribute*

```python
gas: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L158-L158)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.gas_concentration

*attribute* · *instance attribute*

```python
gas_concentration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L159-L159)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.gas_molar_mass

*attribute* · *instance attribute*

```python
gas_molar_mass: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L160-L160)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.gas_vapor_pressure

*attribute* · *instance attribute*

```python
gas_vapor_pressure: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L161-L161)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.request

*attribute* · *instance attribute*

```python
request: ResidentThermodynamicUpdateRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L157-L157)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.thermodynamics

*attribute* · *instance attribute*

```python
thermodynamics: ThermodynamicsConfig
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L166-L166)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicBinding.__init__

*method*

```python
def __init__(request: ResidentThermodynamicUpdateRequest, gas: object, gas_concentration: object, gas_molar_mass: object, gas_vapor_pressure: object, environment_temperature: object, environment_saturation_ratio: object, device: object, dimensions: object, thermodynamics: ThermodynamicsConfig) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L0-L0)

## particula.execution.thermodynamic_updates.PreparedResidentThermodynamicConsumer

*class* · *dataclass*

```python
class PreparedResidentThermodynamicConsumer
```

Bind a current consumer refresh window without changing state.

**Attributes**

- `binding` (`PreparedResidentThermodynamicBinding`) — Validated common thermodynamic writer binding.
- `node` (`ProcessNode`) — Exact consumer node expected at the current cursor.
- `virtual_nodes` (`tuple[ProcessNode, ...]`) — Current preceding virtual refresh nodes.
- `refresh_vapor` (`bool`) — Whether vapor pressure is stale at preparation time.
- `refresh_saturation` (`bool`) — Whether saturation ratio is stale at preparation time.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L169-L186)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicConsumer.binding

*attribute* · *instance attribute*

```python
binding: PreparedResidentThermodynamicBinding
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L182-L182)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicConsumer.node

*attribute* · *instance attribute*

```python
node: ProcessNode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L183-L183)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicConsumer.refresh_saturation

*attribute* · *instance attribute*

```python
refresh_saturation: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L186-L186)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicConsumer.refresh_vapor

*attribute* · *instance attribute*

```python
refresh_vapor: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L185-L185)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicConsumer.virtual_nodes

*attribute* · *instance attribute*

```python
virtual_nodes: tuple[ProcessNode, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L184-L184)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicConsumer.__init__

*method*

```python
def __init__(binding: PreparedResidentThermodynamicBinding, node: ProcessNode, virtual_nodes: tuple[ProcessNode, ...], refresh_vapor: bool, refresh_saturation: bool) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L0-L0)

## particula.execution.thermodynamic_updates.PreparedResidentThermodynamicSequence

*class* · *dataclass*

```python
class PreparedResidentThermodynamicSequence
```

Freeze the two thermodynamic consumer windows for resident enqueue.

The coordinator is used only while constructing this carrier. Enqueue uses
the retained writer inputs and consumer products directly, then records
successful nodes, so it does not validate bindings, resolve nodes, or
construct a coordinator.

**Attributes**

- `binding` (`PreparedResidentThermodynamicBinding`) — Common resident arrays and thermodynamic configuration.
- `condensation` (`PreparedResidentThermodynamicConsumer`) — Prepared refresh window before condensation.
- `diagnostics` (`PreparedResidentThermodynamicConsumer`) — Prepared refresh window before diagnostics.
- `cursor` (`int`) — Number of successful operations recorded during enqueue.
- `stale_states` (`set[InvalidatedState]`) — Derived fields requiring refresh before consumption.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L189-L215)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicSequence.binding

*attribute* · *instance attribute*

```python
binding: PreparedResidentThermodynamicBinding
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L206-L206)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicSequence.condensation

*attribute* · *instance attribute*

```python
condensation: PreparedResidentThermodynamicConsumer
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L207-L207)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicSequence.cursor

*attribute* · *class attribute* · *instance attribute*

```python
cursor: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L209-L209)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicSequence.diagnostics

*attribute* · *instance attribute*

```python
diagnostics: PreparedResidentThermodynamicConsumer
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L208-L208)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicSequence.stale_states

*attribute* · *class attribute* · *instance attribute*

```python
stale_states: set[InvalidatedState] = field(default_factory=lambda : {InvalidatedState.VAPOR_PRESSURE, InvalidatedState.SATURATION_RATIO})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L210-L215)

### particula.execution.thermodynamic_updates.PreparedResidentThermodynamicSequence.__init__

*method*

```python
def __init__(binding: PreparedResidentThermodynamicBinding, condensation: PreparedResidentThermodynamicConsumer, diagnostics: PreparedResidentThermodynamicConsumer, cursor: int = 0, stale_states: set[InvalidatedState] = lambda : {InvalidatedState.VAPOR_PRESSURE, InvalidatedState.SATURATION_RATIO}()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L0-L0)

## particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator

*class*

```python
class ResidentThermodynamicUpdateCoordinator
```

Track derived-state freshness around explicit resident consumer calls.

The coordinator begins with vapor pressure and saturation ratio stale. A
successfully reported ordinary node contributes its declared invalidations.
Before each scheduled condensation or diagnostics callback, virtual refresh
nodes write only stale fields in vapor-pressure then saturation order. A
failed writer or callback does not advance the schedule cursor; a successful
vapor refresh remains fresh if a later saturation refresh fails.

**Attributes**

- `cursor` (`int`) — Number of successfully consumed schedule node IDs.
- `stale_states` (`frozenset[InvalidatedState]`) — Immutable view of coordinator-owned stale derived-state markers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L524-L891)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator._cursor

*attribute* · *instance attribute*

```python
_cursor = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L556-L556)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator._request

*attribute* · *instance attribute*

```python
_request = request
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L555-L555)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator._stale

*attribute* · *instance attribute*

```python
_stale = {InvalidatedState.VAPOR_PRESSURE, InvalidatedState.SATURATION_RATIO}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L557-L560)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator.cursor

*property*

```python
cursor: int
```

Return the number of schedule IDs successfully consumed.

**Returns**

- (`int`) — Number of consumed ordinary, virtual, and consumer nodes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L563-L569)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator.stale_states

*property*

```python
stale_states: frozenset[InvalidatedState]
```

Return coordinator-owned derived-state stale markers.

**Returns**

- (`frozenset[InvalidatedState]`) — Immutable set of fields that require refresh before their next
- (`frozenset[InvalidatedState]`) — supported consumer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L572-L579)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator.__init__

*method*

```python
def __init__(request: ResidentThermodynamicUpdateRequest) -> None
```

Retain a request and initialize both derived fields as stale.

**Parameters**

- `request` (`ResidentThermodynamicUpdateRequest`) — Exact concrete binding to the resident session, registry, resolved graph, schedule, and thermodynamic configuration.

**Raises**

- `TypeError` — If ``request`` is not an exact
``ResidentThermodynamicUpdateRequest``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L540-L560)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator._next_node

*method*

```python
def _next_node() -> ProcessNode
```

Return the next node after complete metadata validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L647-L658)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator._refresh_saturation_ratio

*method*

```python
def _refresh_saturation_ratio() -> None
```

Launch the private resident saturation-ratio writer when nonempty.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L700-L719)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator._refresh_stale_fields

*method*

```python
def _refresh_stale_fields() -> None
```

Write only stale derived fields in canonical dependency order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L848-L855)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator._refresh_vapor_pressure

*method*

```python
def _refresh_vapor_pressure() -> None
```

Delegate the authoritative vapor-pressure writer unchanged.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L691-L698)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator._resolve_consumer_window

*method*

```python
def _resolve_consumer_window() -> tuple[list[ProcessNode], ProcessNode]
```

Return the canonical virtual-writer window and its consumer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L857-L891)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator._validate_binding

*method*

```python
def _validate_binding() -> None
```

Validate resident ownership and exact graph/schedule provenance.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L581-L629)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator._validate_role

*method* · *staticmethod*

```python
def _validate_role(node: ProcessNode) -> None
```

Validate the closed canonical role schema used by this boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L631-L645)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator.execute_consumer

*method*

```python
def execute_consumer(node: ProcessNode, callback: Callable[[], object]) -> object
```

Refresh stale fields and call the next scheduled consumer once.

Only condensation and diagnostics consumers are supported. The method
consumes immediately preceding virtual refresh nodes when present,
writes stale fields in dependency order, then invokes ``callback``.
Stale fields may persist across reported ordinary nodes, so diagnostics
may refresh them without an adjacent virtual-node window. It applies the
consumer invalidations and advances the cursor only after the callback
returns successfully.

**Parameters**

- `node` (`ProcessNode`) — Exact next scheduled condensation or diagnostics graph node.
- `callback` (`Callable[[], object]`) — Zero-argument callable that executes that consumer.

**Returns**

- (`object`) — The value returned by ``callback``.

**Raises**

- `TypeError` — If ``node`` is not an exact ``ProcessNode`` or callback
is not callable.
- `ValueError` — If resident binding, schedule metadata, refresh-window
ordering, or the requested consumer is invalid.
- `Exception` — Propagates a refresh-writer or callback failure without
consuming the consumer node.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L796-L846)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator.prepare_consumer

*method*

```python
def prepare_consumer(binding: PreparedResidentThermodynamicBinding, node: ProcessNode) -> PreparedResidentThermodynamicConsumer
```

Bind the current consumer window without changing coordinator state.

This declaration-consumer seam intentionally reads the coordinator's
current
state on every call. It does not execute writers or consume schedule
entries; the caller must report successful completion separately.

**Parameters**

- `binding` (`PreparedResidentThermodynamicBinding`) — Exact common declaration binding for this coordinator request.
- `node` (`ProcessNode`) — Exact expected condensation or diagnostics consumer node.

**Returns**

- (`PreparedResidentThermodynamicConsumer`) — Immutable current-window binding for enqueue-only refresh writers.

**Raises**

- `TypeError` — If ``binding`` or ``node`` has an unsupported exact type.
- `ValueError` — If the binding drifts, metadata is invalid, or ``node``
is not the current supported consumer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L721-L794)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateCoordinator.record_completed

*method*

```python
def record_completed(node: ProcessNode) -> None
```

Record one successfully completed ordinary schedule node.

The node must be the next scheduled graph member and must not be a
virtual refresh or supported consumer. Its declared invalidations are
added to the coordinator-owned stale markers. Thus completed
communication and volume barriers mark saturation stale only; neither
writes vapor pressure or saturation, launches a refresh, transfers, or
synchronizes.

**Parameters**

- `node` (`ProcessNode`) — Exact next canonical graph node, reported only after its caller-owned execution succeeds.

**Raises**

- `TypeError` — If ``node`` is not an exact ``ProcessNode``.
- `ValueError` — If resident binding or schedule metadata is invalid, or
if ``node`` is out of order, virtual, or a consumer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L660-L689)

## particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateRequest

*class* · *dataclass*

```python
class ResidentThermodynamicUpdateRequest
```

Bind a coordinator to exact resident state and resolved metadata.

Construction intentionally validates only exact concrete carrier types.
Graph provenance, schedule membership, session ownership, and array schemas
are checked immediately before a reported node or consumer is executed.

**Attributes**

- `session` (`ResidentSession`) — Active resident session whose gas and environment fields are refreshed in place.
- `registry` (`object`) — Pinned registry that must be bound to ``session``.
- `graph` (`ResolvedProcessGraph`) — Resolver-produced process graph that owns canonical nodes.
- `schedule` (`ResolvedTimestepSchedule`) — Resolver-produced ordered schedule of graph nodes.
- `thermodynamics` (`ThermodynamicsConfig`) — Configuration passed unchanged to the authoritative vapor-pressure writer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L89-L133)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateRequest.graph

*attribute* · *instance attribute*

```python
graph: ResolvedProcessGraph
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L109-L109)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateRequest.registry

*attribute* · *instance attribute*

```python
registry: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L108-L108)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateRequest.schedule

*attribute* · *instance attribute*

```python
schedule: ResolvedTimestepSchedule
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L110-L110)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateRequest.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L107-L107)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateRequest.thermodynamics

*attribute* · *instance attribute*

```python
thermodynamics: ThermodynamicsConfig
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L111-L111)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateRequest.__init__

*method*

```python
def __init__(session: ResidentSession, registry: object, graph: ResolvedProcessGraph, schedule: ResolvedTimestepSchedule, thermodynamics: ThermodynamicsConfig) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L0-L0)

### particula.execution.thermodynamic_updates.ResidentThermodynamicUpdateRequest.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact retained carrier types without inspecting payloads.

**Raises**

- `TypeError` — If any retained object is not its required exact
concrete type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L113-L133)

## particula.execution.thermodynamic_updates._enqueue_prepared_saturation_ratio

*function*

```python
def _enqueue_prepared_saturation_ratio(prepared: PreparedResidentThermodynamicConsumer) -> None
```

Enqueue the selected bound saturation-ratio refresh only.

This private retained-reference seam does no validation, allocation,
readback, transfer, synchronization, lookup, or rebinding. An empty window
is write-free; rollback is not promised after its writer launches.

**Parameters**

- `prepared` (`PreparedResidentThermodynamicConsumer`) — Current consumer binding that selects the saturation refresh.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L383-L412)

## particula.execution.thermodynamic_updates._enqueue_prepared_vapor_pressure

*function*

```python
def _enqueue_prepared_vapor_pressure(prepared: PreparedResidentThermodynamicConsumer) -> None
```

Enqueue the selected bound vapor-pressure refresh only.

This private retained-reference seam does no validation, allocation,
readback, transfer, synchronization, lookup, or rebinding. An empty window
is write-free; rollback is not promised after its writer launches.

**Parameters**

- `prepared` (`PreparedResidentThermodynamicConsumer`) — Current consumer binding that selects the vapor refresh.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L352-L380)

## particula.execution.thermodynamic_updates._refresh_saturation_ratio_kernel

*function*

```python
def _refresh_saturation_ratio_kernel(concentration: Any, molar_mass: Any, vapor_pressure: Any, temperature: Any, saturation_ratio: Any) -> None
```

Write saturation ratios from resident gas and environment fields.

The kernel applies ``c × R × T / (M × p_v)`` independently to each
box/species lane, where concentration ``c`` is in kg/m³, ``R`` is the gas
constant, temperature ``T`` is in K, molar mass ``M`` is in kg/mol, and
vapor pressure ``p_v`` is in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L54-L75)

## particula.execution.thermodynamic_updates._registry_type

*function*

```python
def _registry_type() -> type[object]
```

Lazily resolve the concrete registry type without a package export.

**Returns**

- (`type[object]`) — The direct-module-only ``GPUResourceRegistry`` class.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L78-L86)

## particula.execution.thermodynamic_updates.record_prepared_thermodynamic_success

*function*

```python
def record_prepared_thermodynamic_success(sequence: PreparedResidentThermodynamicSequence, node: ProcessNode) -> None
```

Advance prepared freshness metadata after one successful node.

The caller invokes this only after the retained writer or process callable
returns. A failed callable consequently leaves the cursor and stale markers
unchanged.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L415-L433)

## particula.execution.thermodynamic_updates.setup_prepared_thermodynamic_binding

*function*

```python
def setup_prepared_thermodynamic_binding(prepared_timestep: object, request: object) -> PreparedResidentThermodynamicBinding
```

Validate a declaration timestep and bind thermodynamic arrays once.

**Parameters**

- `prepared_timestep` (`object`) — Exact declaration timestep retaining the request configuration.
- `request` (`object`) — Exact thermodynamic update request to validate and bind.

**Returns**

- (`PreparedResidentThermodynamicBinding`) — Immutable common binding for later current-cursor consumer preparation.

**Raises**

- `TypeError` — If either carrier has an unsupported exact type.
- `ValueError` — If retained declaration identities, resident ownership,
metadata, or
primary arrays are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L271-L349)

## particula.execution.thermodynamic_updates.setup_prepared_thermodynamic_sequence

*function*

```python
def setup_prepared_thermodynamic_sequence(prepared_timestep: object, request: object, condensation_node: object, diagnostics_node: object) -> PreparedResidentThermodynamicSequence
```

Validate and freeze the condensation and diagnostics windows.

**Parameters**

- `prepared_timestep` (`object`) — Exact READY declaration timestep retaining resident arrays.
- `request` (`object`) — Exact thermodynamic request associated with the timestep.
- `condensation_node` (`object`) — Resolver-produced condensation consumer node.
- `diagnostics_node` (`object`) — Resolver-produced diagnostics consumer node.

**Returns**

- (`PreparedResidentThermodynamicSequence`) — Prepared sequence with both consumer refresh windows retained by
- (`PreparedResidentThermodynamicSequence`) — identity.

**Raises**

- `TypeError` — If a supplied carrier or node has an invalid exact type.
- `ValueError` — If nodes do not belong to the prepared canonical schedule.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/thermodynamic_updates.py#L218-L268)
