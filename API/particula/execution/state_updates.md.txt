# particula.execution.state_updates

*module*

Apply caller-prescribed resident environment and gas replacements.

This direct-import-only module binds immutable update requests to one exact
resident session, resource registry, resolved graph, and graph node. It uses
temporary device scalar validation storage before copying only the designated
resident arrays in place. Canonical empty schemas are intentional write-free
no-ops. It neither schedules, refreshes derived state, transfers host data,
acquires resources, changes lifecycle state, nor provides a fallback. Prepared
setup owns validation and binding; retained-reference enqueue performs no
payload validation, allocation, readback, transfer, synchronization, lookup,
rebinding, or recovery. Setup rejection occurs before a copy writer; rollback
is not promised after a copy writer launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L1-L1)

## particula.execution.state_updates.PreparedResidentEnvironmentUpdate

*class* · *dataclass*

```python
class PreparedResidentEnvironmentUpdate
```

Bind validated environment replacement arrays for enqueue only.

This concrete-only record pins sources and destinations by identity. Its
enqueue path issues only the retained copies, with no validation, lookup,
rebinding, allocation, readback, transfer, or synchronization.

**Attributes**

- `request` (`ResidentEnvironmentUpdateRequest`) — Exact update request validated during setup.
- `target` (`object`) — Resident environment container updated in place.
- `temperature` (`object`) — Validated caller-owned source temperature array in K.
- `pressure` (`object`) — Validated caller-owned source pressure array in Pa.
- `target_temperature` (`object`) — Bound resident temperature destination array.
- `target_pressure` (`object`) — Bound resident pressure destination array.
- `dimensions` (`object`) — Exact resident dimensions used for the empty-schema check.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L161-L185)

### particula.execution.state_updates.PreparedResidentEnvironmentUpdate.dimensions

*attribute* · *instance attribute*

```python
dimensions: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L185-L185)

### particula.execution.state_updates.PreparedResidentEnvironmentUpdate.pressure

*attribute* · *instance attribute*

```python
pressure: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L182-L182)

### particula.execution.state_updates.PreparedResidentEnvironmentUpdate.request

*attribute* · *instance attribute*

```python
request: ResidentEnvironmentUpdateRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L179-L179)

### particula.execution.state_updates.PreparedResidentEnvironmentUpdate.target

*attribute* · *instance attribute*

```python
target: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L180-L180)

### particula.execution.state_updates.PreparedResidentEnvironmentUpdate.target_pressure

*attribute* · *instance attribute*

```python
target_pressure: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L184-L184)

### particula.execution.state_updates.PreparedResidentEnvironmentUpdate.target_temperature

*attribute* · *instance attribute*

```python
target_temperature: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L183-L183)

### particula.execution.state_updates.PreparedResidentEnvironmentUpdate.temperature

*attribute* · *instance attribute*

```python
temperature: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L181-L181)

### particula.execution.state_updates.PreparedResidentEnvironmentUpdate.__init__

*method*

```python
def __init__(request: ResidentEnvironmentUpdateRequest, target: object, temperature: object, pressure: object, target_temperature: object, target_pressure: object, dimensions: object) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L0-L0)

## particula.execution.state_updates.PreparedResidentGasUpdate

*class* · *dataclass*

```python
class PreparedResidentGasUpdate
```

Bind validated gas replacement arrays for enqueue only.

This concrete-only record pins sources and destinations by identity. Its
enqueue path issues only the retained copy, with no validation, lookup,
rebinding, allocation, readback, transfer, or synchronization.

**Attributes**

- `request` (`ResidentGasUpdateRequest`) — Exact update request validated during setup.
- `target` (`object`) — Resident gas container updated in place.
- `concentration` (`object`) — Validated caller-owned gas concentration source array.
- `target_concentration` (`object`) — Bound resident concentration destination array.
- `dimensions` (`object`) — Exact resident dimensions used for the empty-schema check.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L188-L208)

### particula.execution.state_updates.PreparedResidentGasUpdate.concentration

*attribute* · *instance attribute*

```python
concentration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L206-L206)

### particula.execution.state_updates.PreparedResidentGasUpdate.dimensions

*attribute* · *instance attribute*

```python
dimensions: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L208-L208)

### particula.execution.state_updates.PreparedResidentGasUpdate.request

*attribute* · *instance attribute*

```python
request: ResidentGasUpdateRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L204-L204)

### particula.execution.state_updates.PreparedResidentGasUpdate.target

*attribute* · *instance attribute*

```python
target: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L205-L205)

### particula.execution.state_updates.PreparedResidentGasUpdate.target_concentration

*attribute* · *instance attribute*

```python
target_concentration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L207-L207)

### particula.execution.state_updates.PreparedResidentGasUpdate.__init__

*method*

```python
def __init__(request: ResidentGasUpdateRequest, target: object, concentration: object, target_concentration: object, dimensions: object) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L0-L0)

## particula.execution.state_updates.ResidentEnvironmentUpdateRequest

*class* · *dataclass*

```python
class ResidentEnvironmentUpdateRequest
```

Retain one exact graph-bound environment replacement request.

The immutable carrier preserves session, registry, graph, node, and Warp
input-array identities. Construction validates only the concrete dependency
carriers; execution validates the graph binding, array schemas, ownership,
and positive finite temperature and pressure values.

**Attributes**

- `session` (`ResidentSession`) — Exact resident session whose environment arrays may be updated.
- `registry` (`GPUResourceRegistry`) — Exact registry pinned to ``session``.
- `graph` (`ResolvedProcessGraph`) — Exact resolved graph containing ``node`` by identity.
- `node` (`ProcessNode`) — Canonical ``environment_update`` graph node.
- `temperature` (`object`) — Caller-owned float64 Warp array shaped ``(n_boxes,)`` in K.
- `pressure` (`object`) — Caller-owned float64 Warp array shaped ``(n_boxes,)`` in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L80-L118)

### particula.execution.state_updates.ResidentEnvironmentUpdateRequest.graph

*attribute* · *instance attribute*

```python
graph: ResolvedProcessGraph
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L100-L100)

### particula.execution.state_updates.ResidentEnvironmentUpdateRequest.node

*attribute* · *instance attribute*

```python
node: ProcessNode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L101-L101)

### particula.execution.state_updates.ResidentEnvironmentUpdateRequest.pressure

*attribute* · *instance attribute*

```python
pressure: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L103-L103)

### particula.execution.state_updates.ResidentEnvironmentUpdateRequest.registry

*attribute* · *instance attribute*

```python
registry: GPUResourceRegistry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L99-L99)

### particula.execution.state_updates.ResidentEnvironmentUpdateRequest.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L98-L98)

### particula.execution.state_updates.ResidentEnvironmentUpdateRequest.temperature

*attribute* · *instance attribute*

```python
temperature: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L102-L102)

### particula.execution.state_updates.ResidentEnvironmentUpdateRequest.__init__

*method*

```python
def __init__(session: ResidentSession, registry: GPUResourceRegistry, graph: ResolvedProcessGraph, node: ProcessNode, temperature: object, pressure: object) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L0-L0)

### particula.execution.state_updates.ResidentEnvironmentUpdateRequest.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate only exact dependency carrier types in field order.

**Raises**

- `TypeError` — If a dependency is not its exact required concrete type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L105-L118)

## particula.execution.state_updates.ResidentGasUpdateRequest

*class* · *dataclass*

```python
class ResidentGasUpdateRequest
```

Retain one exact graph-bound gas concentration replacement request.

The immutable carrier preserves session, registry, graph, node, and Warp
input-array identities. Construction validates only the concrete dependency
carriers; execution validates the graph binding, array schema, ownership,
and finite nonnegative concentration values.

**Attributes**

- `session` (`ResidentSession`) — Exact resident session whose gas concentration may be updated.
- `registry` (`GPUResourceRegistry`) — Exact registry pinned to ``session``.
- `graph` (`ResolvedProcessGraph`) — Exact resolved graph containing ``node`` by identity.
- `node` (`ProcessNode`) — Canonical ``gas_update`` graph node.
- `concentration` (`object`) — Caller-owned float64 Warp array shaped ``(n_boxes, n_species)`` in the resident gas concentration units.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L121-L158)

### particula.execution.state_updates.ResidentGasUpdateRequest.concentration

*attribute* · *instance attribute*

```python
concentration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L143-L143)

### particula.execution.state_updates.ResidentGasUpdateRequest.graph

*attribute* · *instance attribute*

```python
graph: ResolvedProcessGraph
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L141-L141)

### particula.execution.state_updates.ResidentGasUpdateRequest.node

*attribute* · *instance attribute*

```python
node: ProcessNode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L142-L142)

### particula.execution.state_updates.ResidentGasUpdateRequest.registry

*attribute* · *instance attribute*

```python
registry: GPUResourceRegistry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L140-L140)

### particula.execution.state_updates.ResidentGasUpdateRequest.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L139-L139)

### particula.execution.state_updates.ResidentGasUpdateRequest.__init__

*method*

```python
def __init__(session: ResidentSession, registry: GPUResourceRegistry, graph: ResolvedProcessGraph, node: ProcessNode, concentration: object) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L0-L0)

### particula.execution.state_updates.ResidentGasUpdateRequest.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate only exact dependency carrier types in field order.

**Raises**

- `TypeError` — If a dependency is not its exact required concrete type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L145-L158)

## particula.execution.state_updates.ResidentStateUpdateExecutor

*class*

```python
class ResidentStateUpdateExecutor
```

Apply one graph-bound resident replacement without scheduler behavior.

Execution preserves all resident container and primary-array identities. It
updates only environment temperature and pressure or gas concentration;
canonical empty schemas are successful write-free no-ops. It does not
refresh derived state, schedule work, transfer host data, alter lifecycle
state, or provide rollback after a copy writer launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L611-L714)

### particula.execution.state_updates.ResidentStateUpdateExecutor._execute_environment

*method* · *staticmethod*

```python
def _execute_environment(request: ResidentEnvironmentUpdateRequest) -> object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L644-L686)

### particula.execution.state_updates.ResidentStateUpdateExecutor._execute_gas

*method* · *staticmethod*

```python
def _execute_gas(request: ResidentGasUpdateRequest) -> object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L688-L714)

### particula.execution.state_updates.ResidentStateUpdateExecutor.execute

*method*

```python
def execute(request: object) -> object
```

Validate and apply one exact environment or gas update request.

**Parameters**

- `request` (`object`) — Exact immutable environment or gas update request.

**Returns**

- (`object`) — The identical resident environment or gas container that was
- (`object`) — targeted by ``request``.

**Raises**

- `TypeError` — If ``request`` is not an exact supported request type.
- `ValueError` — If the pinned binding, graph role, input schema,
ownership, or scalar values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L621-L642)

## particula.execution.state_updates._array_range

*function*

```python
def _array_range(value: Any, shape: tuple[int, ...], name: str, item_size: int = 8) -> tuple[int, int] | None
```

Validate contiguous metadata and return a nonempty half-open range.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L211-L246)

## particula.execution.state_updates._enqueue_prepared_environment_update

*function*

```python
def _enqueue_prepared_environment_update(prepared: PreparedResidentEnvironmentUpdate) -> object
```

Enqueue the bound temperature-then-pressure copies only.

The setup-validated binding is retained by identity. This concrete private
seam performs neither payload validation nor allocation, readback, transfer,
synchronization, lookup, or rebinding. Empty schemas are write-free;
rollback is not promised after a copy writer launches.

**Parameters**

- `prepared` (`PreparedResidentEnvironmentUpdate`) — Previously validated immutable environment copy binding.

**Returns**

- (`object`) — The identical resident environment container.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L560-L585)

## particula.execution.state_updates._enqueue_prepared_gas_update

*function*

```python
def _enqueue_prepared_gas_update(prepared: PreparedResidentGasUpdate) -> object
```

Enqueue the bound gas concentration copy only.

The setup-validated binding is retained by identity. This concrete private
seam performs neither payload validation nor allocation, readback, transfer,
synchronization, lookup, or rebinding. Empty schemas are write-free;
rollback is not promised after a copy writer launches.

**Parameters**

- `prepared` (`PreparedResidentGasUpdate`) — Previously validated immutable gas copy binding.

**Returns**

- (`object`) — The identical resident gas container.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L588-L608)

## particula.execution.state_updates._overlaps

*function*

```python
def _overlaps(left: tuple[int, int] | None, right: tuple[int, int] | None) -> bool
```

Return whether two nonempty half-open byte ranges overlap.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L266-L275)

## particula.execution.state_updates._primary_arrays

*function*

```python
def _primary_arrays(session: ResidentSession) -> tuple[Any, ...]
```

Return the protected resident primary arrays in canonical order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L278-L296)

## particula.execution.state_updates._registry_type

*function*

```python
def _registry_type() -> type[object]
```

Lazily resolve the concrete registry type for request validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L73-L77)

## particula.execution.state_updates._reject_primary_aliases

*function*

```python
def _reject_primary_aliases(session: ResidentSession, values: tuple[object, ...], ranges: tuple[tuple[int, int] | None, ...]) -> None
```

Reject input identities and nonempty ranges shared with primaries.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L299-L324)

## particula.execution.state_updates._scan

*function*

```python
def _scan(value: Any, size: int, *, positive: bool, name: str, matrix_shape: tuple[int, int] | None = None) -> None
```

Scan a nonempty device payload with transient scalar validation state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L375-L399)

## particula.execution.state_updates._scan_nonnegative

*function*

```python
def _scan_nonnegative(values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32))
```

Mark invalid entries that are nonfinite or negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L51-L59)

## particula.execution.state_updates._scan_nonnegative_matrix

*function*

```python
def _scan_nonnegative_matrix(values: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32))
```

Mark invalid matrix entries that are nonfinite or negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L62-L70)

## particula.execution.state_updates._scan_positive

*function*

```python
def _scan_positive(values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32))
```

Mark invalid entries that are nonfinite or not strictly positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L40-L48)

## particula.execution.state_updates._validate_array

*function*

```python
def _validate_array(value: object, shape: tuple[int, ...], device: object, name: str) -> tuple[int, int] | None
```

Validate one caller-prescribed float64 Warp array and its byte range.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L249-L263)

## particula.execution.state_updates._validate_node

*function*

```python
def _validate_node(request: object, *, environment: bool) -> None
```

Validate resolver provenance, membership, and canonical update role.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L327-L372)

## particula.execution.state_updates._validate_prepared_binding

*function*

```python
def _validate_prepared_binding(prepared_timestep: object, request: object, *, environment: bool) -> None
```

Validate the declaration binding before retaining copy identities.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L402-L450)

## particula.execution.state_updates.setup_prepared_environment_update

*function*

```python
def setup_prepared_environment_update(prepared_timestep: object, request: object) -> PreparedResidentEnvironmentUpdate
```

Validate and bind an environment replacement for later enqueue.

**Parameters**

- `prepared_timestep` (`object`) — Exact declaration timestep retaining ``request`` by identity.
- `request` (`object`) — Exact environment update request to validate and bind.

**Returns**

- (`PreparedResidentEnvironmentUpdate`) — Immutable copy binding whose enqueue phase copies temperature then
- (`PreparedResidentEnvironmentUpdate`) — pressure without repeating validation.

**Raises**

- `TypeError` — If either carrier has an unsupported exact type.
- `ValueError` — If declaration identities, source metadata, aliases, or
values are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L453-L513)

## particula.execution.state_updates.setup_prepared_gas_update

*function*

```python
def setup_prepared_gas_update(prepared_timestep: object, request: object) -> PreparedResidentGasUpdate
```

Validate and bind a gas replacement for later enqueue.

**Parameters**

- `prepared_timestep` (`object`) — Exact declaration timestep retaining ``request`` by identity.
- `request` (`object`) — Exact gas update request to validate and bind.

**Returns**

- (`PreparedResidentGasUpdate`) — Immutable copy binding whose enqueue phase copies concentration without
- (`PreparedResidentGasUpdate`) — repeating validation.

**Raises**

- `TypeError` — If either carrier has an unsupported exact type.
- `ValueError` — If declaration identities, source metadata, aliases, or
values are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/state_updates.py#L516-L557)
