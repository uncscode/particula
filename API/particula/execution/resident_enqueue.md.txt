# particula.execution.resident_enqueue

*module*

Prepare READY resident graph metadata for later enqueue phases.

This concrete direct-import-only declaration boundary performs validation and
freezes
READY-state identity metadata only. It resolves the registry's
already-published capture resource set and cached report by identity, but does
not construct executors,
Capture, enqueue, dispatch, acquire resources, inspect payloads, transfer,
synchronize, mutate a lifecycle, or fall back. Preparation retains the exact
READY binding but does not authorize token entry or dispatch; identity or
lifecycle drift requires a new prepared record and rejects before token entry.
It performs no allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L1-L1)

## particula.execution.resident_enqueue.PreparedResidentTimestep

*class* · *dataclass*

```python
class PreparedResidentTimestep
```

Retain exact READY-bound metadata for a future resident enqueue.

This frozen, identity-semantic carrier is preparation output only. It
retains existing host metadata and published resources without copying or
acquiring them, and it does not authorize capture, enqueue, or dispatch.

**Attributes**

- `request` (`'ResidentSimulationRequest'`) — Exact complete resident request retained by the binding.
- `binding` (`'ResidentGraphCaptureBinding'`) — Exact READY graph-capture binding for ``request``.
- `lifecycle` (`'GraphCaptureLifecycle'`) — Exact READY lifecycle retained by ``binding``.
- `signature` (`'ResidentGraphCaptureSignature'`) — Exact lifecycle signature used for compatibility checks.
- `session` (`object`) — Exact resident session shared by request and binding.
- `registry` (`'GPUResourceRegistry'`) — Exact resource registry shared by request and binding.
- `guard` (`object`) — Exact closed resident-step guard shared by request and binding.
- `device` (`object`) — Exact signature device.
- `dimensions` (`object`) — Exact signature resident dimensions.
- `graph` (`object`) — Exact resolved graph retained by ``request``.
- `schedule` (`'ResolvedTimestepSchedule'`) — Exact resolved schedule retained by ``request``.
- `ordered_node_ids` (`tuple[object, ...]`) — Exact canonical schedule node-ID tuple.
- `duration` (`object`) — Original finite, nonnegative timestep duration.
- `primary_arrays` (`tuple[object, ...]`) — Exact signature primary-array tuple.
- `resource_views` (`tuple[object, ...]`) — Exact signature published-resource tuple.
- `capture_requirements` (`object`) — Exact request requirements retained by identity.
- `capture_set` (`object`) — Exact registry-published capture set.
- `capture_report` (`object`) — Exact immutable logical-byte report from the set.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L113-L266)

### particula.execution.resident_enqueue.PreparedResidentTimestep.binding

*attribute* · *instance attribute*

```python
binding: 'ResidentGraphCaptureBinding'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L143-L143)

### particula.execution.resident_enqueue.PreparedResidentTimestep.capture_report

*attribute* · *class attribute* · *instance attribute*

```python
capture_report: object = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L159-L159)

### particula.execution.resident_enqueue.PreparedResidentTimestep.capture_requirements

*attribute* · *class attribute* · *instance attribute*

```python
capture_requirements: object = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L157-L157)

### particula.execution.resident_enqueue.PreparedResidentTimestep.capture_set

*attribute* · *class attribute* · *instance attribute*

```python
capture_set: object = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L158-L158)

### particula.execution.resident_enqueue.PreparedResidentTimestep.device

*attribute* · *instance attribute*

```python
device: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L149-L149)

### particula.execution.resident_enqueue.PreparedResidentTimestep.dimensions

*attribute* · *instance attribute*

```python
dimensions: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L150-L150)

### particula.execution.resident_enqueue.PreparedResidentTimestep.duration

*attribute* · *instance attribute*

```python
duration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L154-L154)

### particula.execution.resident_enqueue.PreparedResidentTimestep.graph

*attribute* · *instance attribute*

```python
graph: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L151-L151)

### particula.execution.resident_enqueue.PreparedResidentTimestep.guard

*attribute* · *instance attribute*

```python
guard: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L148-L148)

### particula.execution.resident_enqueue.PreparedResidentTimestep.lifecycle

*attribute* · *instance attribute*

```python
lifecycle: 'GraphCaptureLifecycle'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L144-L144)

### particula.execution.resident_enqueue.PreparedResidentTimestep.ordered_node_ids

*attribute* · *instance attribute*

```python
ordered_node_ids: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L153-L153)

### particula.execution.resident_enqueue.PreparedResidentTimestep.primary_arrays

*attribute* · *instance attribute*

```python
primary_arrays: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L155-L155)

### particula.execution.resident_enqueue.PreparedResidentTimestep.registry

*attribute* · *instance attribute*

```python
registry: 'GPUResourceRegistry'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L147-L147)

### particula.execution.resident_enqueue.PreparedResidentTimestep.request

*attribute* · *instance attribute*

```python
request: 'ResidentSimulationRequest'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L142-L142)

### particula.execution.resident_enqueue.PreparedResidentTimestep.resource_views

*attribute* · *instance attribute*

```python
resource_views: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L156-L156)

### particula.execution.resident_enqueue.PreparedResidentTimestep.schedule

*attribute* · *instance attribute*

```python
schedule: 'ResolvedTimestepSchedule'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L152-L152)

### particula.execution.resident_enqueue.PreparedResidentTimestep.session

*attribute* · *instance attribute*

```python
session: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L146-L146)

### particula.execution.resident_enqueue.PreparedResidentTimestep.signature

*attribute* · *instance attribute*

```python
signature: 'ResidentGraphCaptureSignature'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L145-L145)

### particula.execution.resident_enqueue.PreparedResidentTimestep.__init__

*method*

```python
def __init__(request: 'ResidentSimulationRequest', binding: 'ResidentGraphCaptureBinding', lifecycle: 'GraphCaptureLifecycle', signature: 'ResidentGraphCaptureSignature', session: object, registry: 'GPUResourceRegistry', guard: object, device: object, dimensions: object, graph: object, schedule: 'ResolvedTimestepSchedule', ordered_node_ids: tuple[object, ...], duration: object, primary_arrays: tuple[object, ...], resource_views: tuple[object, ...], capture_requirements: object = None, capture_set: object = None, capture_report: object = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L0-L0)

### particula.execution.resident_enqueue.PreparedResidentTimestep.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact carriers, READY state, and retained identities.

**Raises**

- `TypeError` — If a carrier, tuple, or duration has an invalid type.
- `ValueError` — If duration is invalid, identities drift, or lifecycle
state is not READY.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L161-L266)

## particula.execution.resident_enqueue._graph_capture_types

*function*

```python
def _graph_capture_types() -> tuple[type[object], type[object], type[object]]
```

Return concrete graph-capture carrier types lazily.

**Returns**

- (`tuple[type[object], type[object], type[object]]`) — Exact binding, lifecycle, and signature types in that order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L45-L61)

## particula.execution.resident_enqueue._request_type

*function*

```python
def _request_type() -> type[object]
```

Return the concrete resident simulation request type lazily.

**Returns**

- (`type[object]`) — The direct-import-only resident simulation request type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L34-L42)

## particula.execution.resident_enqueue._validate_ready_attachment

*function*

```python
def _validate_ready_attachment(request: object, binding: object, lifecycle: object, signature: object, session: object, registry: object, guard: object) -> None
```

Require one exact, closed, active READY resident attachment.

This check is shared by direct prepared-carrier construction and preparation
return guards so neither path can retain stale attachment metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L64-L110)

## particula.execution.resident_enqueue.prepare_resident_timestep

*function*

```python
def prepare_resident_timestep(request: object, duration: object) -> PreparedResidentTimestep
```

Validate and freeze one READY resident timestep without side effects.

The direct-only preparation boundary performs shared read-only metadata
    validation, cached capture-publication lookup, and identity retention in
    a frozen carrier. It does not construct executors, open a guard token,
    acquire resources, inspect payloads, capture, enqueue, dispatch,
    transfer, synchronize, mutate lifecycle state, or fall back.

**Parameters**

- `request` (`object`) — Exact attached resident simulation request.
- `duration` (`object`) — Non-boolean finite, nonnegative timestep duration.

**Returns**

- (`PreparedResidentTimestep`) — A frozen identity-only prepared timestep.

**Raises**

- `TypeError` — If a request, attachment, retained carrier, or duration has
an inexact or invalid type.
- `ValueError` — If duration, ownership, lifecycle state, signature, or
complete-loop metadata is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/resident_enqueue.py#L269-L396)
