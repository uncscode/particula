# particula.execution.graph_capture

*module*

Qualify, capture, and replay prepared GPU-resident timestep graphs.

This concrete, direct-import-only authority resolves caller-provided
graph-capture support and records identity-based compatibility for an
already-built resident request. Qualification retains a native callable
vocabulary for one exact READY binding. Capture calls native begin, frozen
operation dispatch, and end in order; it issues an authentic record for the
resulting opaque handle in private provenance and releases it after
post-capture
 rejection or lifecycle teardown. Replay accepts only that issued record,
 revalidates its exact binding metadata, and performs one guard-token/native-
 launch/completion path.

This module neither exports a package-level API nor inspects resident payloads.
Replay performs no prepared-host dispatch, allocation, runtime probe, transfer,
readback, synchronization, resource work, RNG initialization or reset,
 fallback, retry, or automatic recapture. This module alone owns issued-handle
 storage with live/releasing/released tombstones and releases handles exactly
 once during lifecycle teardown. Native callbacks run outside the global
 provenance lock, while per-record and per-session leases serialize launch with
 release and terminal lifecycle intent. A native
 launch or completion failure is writer-capable: existing resident cleanup and
 fault classification apply, with no rollback guarantee.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1-L1)

## particula.execution.graph_capture._ACTIVE_CAPTURE_BINDING_IDS

*attribute* · *module attribute*

```python
_ACTIVE_CAPTURE_BINDING_IDS: set[int] = set()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L37-L37)

## particula.execution.graph_capture._ACTIVE_CAPTURE_LOCK

*attribute* · *module attribute*

```python
_ACTIVE_CAPTURE_LOCK = Lock()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L38-L38)

## particula.execution.graph_capture._ISSUED_CAPTURED_GRAPHS

*attribute* · *module attribute*

```python
_ISSUED_CAPTURED_GRAPHS: dict[CapturedResidentGraph, _NativeGraphRecord] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1807-L1807)

## particula.execution.graph_capture._ISSUED_CAPTURED_GRAPHS_CONDITION

*attribute* · *module attribute*

```python
_ISSUED_CAPTURED_GRAPHS_CONDITION = Condition(_ISSUED_CAPTURED_GRAPHS_LOCK)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L43-L43)

## particula.execution.graph_capture._ISSUED_CAPTURED_GRAPHS_LOCK

*attribute* · *module attribute*

```python
_ISSUED_CAPTURED_GRAPHS_LOCK = RLock()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L42-L42)

## particula.execution.graph_capture._NATIVE_HANDLE_PROVENANCE

*attribute* · *module attribute*

```python
_NATIVE_HANDLE_PROVENANCE: dict[int, _NativeGraphRecord] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1808-L1808)

## particula.execution.graph_capture.CapturedResidentGraph

*class* · *dataclass*

```python
class CapturedResidentGraph
```

Retain one completed native capture and its exact resident binding.

Authentic records are issued only after capture completes and publishes
CAPTURED metadata. The opaque ``handle`` is retained by identity only;
capture
rejects only a literal ``None`` handle, while replay compares provenance and
forwards it by identity without inspecting its type, equality, truthiness,
or serialization. Capture owns native begin/end/release work; replay may
only
forward an authentic handle to ``capture_launch`` under one guard token.
This direct-module-only carrier does not expose cleanup or a public export.

**Attributes**

- `qualification` (`PreparedGraphCaptureQualification`) — Exact READY qualification validated before and after native capture.
- `binding` (`ResidentGraphCaptureBinding`) — Exact attached resident graph-capture binding.
- `lifecycle` (`GraphCaptureLifecycle`) — Binding-owned CAPTURED lifecycle successor.
- `signature` (`ResidentGraphCaptureSignature`) — Exact structural signature retained by ``qualification``.
- `request` (`object`) — Exact resident simulation request.
- `session` (`object`) — Exact active resident session.
- `registry` (`object`) — Exact resource registry pinned to ``session``.
- `guard` (`object`) — Exact closed resident step guard.
- `prepared` (`object`) — Exact prepared resident simulation dispatched during capture.
- `timestep` (`object`) — Exact prepared timestep retained by ``prepared``.
- `capture_requirements` (`object`) — Exact published capture-resource requirements.
- `capture_set` (`object`) — Exact published capture-resource set.
- `capture_report` (`object`) — Exact cached logical-byte report for ``capture_set``.
- `device` (`Device`) — Exact qualified non-CPU Warp device. The opaque native handle remains private to module-owned provenance.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1708-L1781)

### particula.execution.graph_capture.CapturedResidentGraph.binding

*attribute* · *instance attribute*

```python
binding: ResidentGraphCaptureBinding
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1742-L1742)

### particula.execution.graph_capture.CapturedResidentGraph.capture_report

*attribute* · *instance attribute*

```python
capture_report: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1753-L1753)

### particula.execution.graph_capture.CapturedResidentGraph.capture_requirements

*attribute* · *instance attribute*

```python
capture_requirements: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1751-L1751)

### particula.execution.graph_capture.CapturedResidentGraph.capture_set

*attribute* · *instance attribute*

```python
capture_set: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1752-L1752)

### particula.execution.graph_capture.CapturedResidentGraph.device

*attribute* · *instance attribute*

```python
device: Device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1754-L1754)

### particula.execution.graph_capture.CapturedResidentGraph.guard

*attribute* · *instance attribute*

```python
guard: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1748-L1748)

### particula.execution.graph_capture.CapturedResidentGraph.lifecycle

*attribute* · *instance attribute*

```python
lifecycle: GraphCaptureLifecycle
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1743-L1743)

### particula.execution.graph_capture.CapturedResidentGraph.prepared

*attribute* · *instance attribute*

```python
prepared: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1749-L1749)

### particula.execution.graph_capture.CapturedResidentGraph.qualification

*attribute* · *instance attribute*

```python
qualification: PreparedGraphCaptureQualification
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1741-L1741)

### particula.execution.graph_capture.CapturedResidentGraph.registry

*attribute* · *instance attribute*

```python
registry: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1747-L1747)

### particula.execution.graph_capture.CapturedResidentGraph.request

*attribute* · *instance attribute*

```python
request: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1745-L1745)

### particula.execution.graph_capture.CapturedResidentGraph.session

*attribute* · *instance attribute*

```python
session: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1746-L1746)

### particula.execution.graph_capture.CapturedResidentGraph.signature

*attribute* · *instance attribute*

```python
signature: ResidentGraphCaptureSignature
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1744-L1744)

### particula.execution.graph_capture.CapturedResidentGraph.timestep

*attribute* · *instance attribute*

```python
timestep: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1750-L1750)

### particula.execution.graph_capture.CapturedResidentGraph.__init__

*method*

```python
def __init__(qualification: PreparedGraphCaptureQualification, binding: ResidentGraphCaptureBinding, lifecycle: GraphCaptureLifecycle, signature: ResidentGraphCaptureSignature, request: object, session: object, registry: object, guard: object, prepared: object, timestep: object, capture_requirements: object, capture_set: object, capture_report: object, device: Device) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L0-L0)

### particula.execution.graph_capture.CapturedResidentGraph.__post_init__

*method*

```python
def __post_init__() -> None
```

Require the exact captured successor and qualification identities.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1756-L1781)

## particula.execution.graph_capture.GraphCaptureAvailability

*class*

```python
class GraphCaptureAvailability(str, Enum)
```

Bases: `str`, `Enum`

Enumerate graph-capture availability outcomes from a lazy probe.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L72-L80)

### particula.execution.graph_capture.GraphCaptureAvailability.AVAILABLE

*attribute* · *class attribute* · *instance attribute*

```python
AVAILABLE = 'available'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L80-L80)

### particula.execution.graph_capture.GraphCaptureAvailability.UNAVAILABLE_DEVICE

*attribute* · *class attribute* · *instance attribute*

```python
UNAVAILABLE_DEVICE = 'unavailable_device'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L78-L78)

### particula.execution.graph_capture.GraphCaptureAvailability.UNAVAILABLE_RUNTIME

*attribute* · *class attribute* · *instance attribute*

```python
UNAVAILABLE_RUNTIME = 'unavailable_runtime'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L77-L77)

### particula.execution.graph_capture.GraphCaptureAvailability.UNSUPPORTED_API

*attribute* · *class attribute* · *instance attribute*

```python
UNSUPPORTED_API = 'unsupported_api'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L79-L79)

### particula.execution.graph_capture.GraphCaptureAvailability.UNSUPPORTED_CPU

*attribute* · *class attribute* · *instance attribute*

```python
UNSUPPORTED_CPU = 'unsupported_cpu'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L75-L75)

### particula.execution.graph_capture.GraphCaptureAvailability.UNSUPPORTED_WARP_CPU

*attribute* · *class attribute* · *instance attribute*

```python
UNSUPPORTED_WARP_CPU = 'unsupported_warp_cpu'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L76-L76)

## particula.execution.graph_capture.GraphCaptureCapability

*class* · *dataclass*

```python
class GraphCaptureCapability
```

Retain an exact device declaration and its capture availability.

**Attributes**

- `device` (`Device`) — Exact device declaration assessed by the resolver.
- `availability` (`GraphCaptureAvailability`) — Availability outcome for ``device``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L83-L107)

### particula.execution.graph_capture.GraphCaptureCapability.availability

*attribute* · *instance attribute*

```python
availability: GraphCaptureAvailability
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L93-L93)

### particula.execution.graph_capture.GraphCaptureCapability.device

*attribute* · *instance attribute*

```python
device: Device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L92-L92)

### particula.execution.graph_capture.GraphCaptureCapability.__init__

*method*

```python
def __init__(device: Device, availability: GraphCaptureAvailability) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L0-L0)

### particula.execution.graph_capture.GraphCaptureCapability.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact device and availability declaration types.

**Raises**

- `TypeError` — If either declaration does not have its required exact
type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L95-L107)

## particula.execution.graph_capture.GraphCaptureCompatibility

*class* · *dataclass*

```python
class GraphCaptureCompatibility
```

State whether a request remains compatible with a signature.

**Attributes**

- `compatible` (`bool`) — Whether every tracked group retains object identity.
- `reason` (`GraphCaptureDriftReason | None`) — First changed group, or ``None`` only when compatible.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L295-L327)

### particula.execution.graph_capture.GraphCaptureCompatibility.compatible

*attribute* · *instance attribute*

```python
compatible: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L304-L304)

### particula.execution.graph_capture.GraphCaptureCompatibility.reason

*attribute* · *instance attribute*

```python
reason: GraphCaptureDriftReason | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L305-L305)

### particula.execution.graph_capture.GraphCaptureCompatibility.__init__

*method*

```python
def __init__(compatible: bool, reason: GraphCaptureDriftReason | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L0-L0)

### particula.execution.graph_capture.GraphCaptureCompatibility.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact types and the compatible/reason invariant.

**Raises**

- `TypeError` — If a field does not have its required exact type.
- `ValueError` — If ``reason`` is absent or present inconsistently with
``compatible``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L307-L327)

## particula.execution.graph_capture.GraphCaptureDriftReason

*class*

```python
class GraphCaptureDriftReason(str, Enum)
```

Bases: `str`, `Enum`

Enumerate signature groups in deterministic first-drift order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L276-L292)

### particula.execution.graph_capture.GraphCaptureDriftReason.COMMUNICATION

*attribute* · *class attribute* · *instance attribute*

```python
COMMUNICATION = 'communication'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L290-L290)

### particula.execution.graph_capture.GraphCaptureDriftReason.CONFIGURATIONS

*attribute* · *class attribute* · *instance attribute*

```python
CONFIGURATIONS = 'configurations'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L291-L291)

### particula.execution.graph_capture.GraphCaptureDriftReason.DEVICE

*attribute* · *class attribute* · *instance attribute*

```python
DEVICE = 'device'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L281-L281)

### particula.execution.graph_capture.GraphCaptureDriftReason.DIAGNOSTICS

*attribute* · *class attribute* · *instance attribute*

```python
DIAGNOSTICS = 'diagnostics'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L289-L289)

### particula.execution.graph_capture.GraphCaptureDriftReason.DIMENSIONS

*attribute* · *class attribute* · *instance attribute*

```python
DIMENSIONS = 'dimensions'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L282-L282)

### particula.execution.graph_capture.GraphCaptureDriftReason.GRAPH

*attribute* · *class attribute* · *instance attribute*

```python
GRAPH = 'graph'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L286-L286)

### particula.execution.graph_capture.GraphCaptureDriftReason.PRIMARY_ARRAYS

*attribute* · *class attribute* · *instance attribute*

```python
PRIMARY_ARRAYS = 'primary_arrays'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L284-L284)

### particula.execution.graph_capture.GraphCaptureDriftReason.PRIMARY_CONTAINERS

*attribute* · *class attribute* · *instance attribute*

```python
PRIMARY_CONTAINERS = 'primary_containers'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L283-L283)

### particula.execution.graph_capture.GraphCaptureDriftReason.REQUEST

*attribute* · *class attribute* · *instance attribute*

```python
REQUEST = 'request'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L279-L279)

### particula.execution.graph_capture.GraphCaptureDriftReason.RESOURCE_VIEWS

*attribute* · *class attribute* · *instance attribute*

```python
RESOURCE_VIEWS = 'resource_views'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L285-L285)

### particula.execution.graph_capture.GraphCaptureDriftReason.RNG_RESOURCES

*attribute* · *class attribute* · *instance attribute*

```python
RNG_RESOURCES = 'rng_resources'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L292-L292)

### particula.execution.graph_capture.GraphCaptureDriftReason.SCHEDULE

*attribute* · *class attribute* · *instance attribute*

```python
SCHEDULE = 'schedule'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L287-L287)

### particula.execution.graph_capture.GraphCaptureDriftReason.SCHEDULE_ORDER

*attribute* · *class attribute* · *instance attribute*

```python
SCHEDULE_ORDER = 'schedule_order'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L288-L288)

### particula.execution.graph_capture.GraphCaptureDriftReason.SESSION

*attribute* · *class attribute* · *instance attribute*

```python
SESSION = 'session'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L280-L280)

## particula.execution.graph_capture.GraphCaptureFailureClassification

*class*

```python
class GraphCaptureFailureClassification(str, Enum)
```

Bases: `str`, `Enum`

Classify whether a host-recorded failure may follow a writer launch.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1097-L1101)

### particula.execution.graph_capture.GraphCaptureFailureClassification.READ_ONLY

*attribute* · *class attribute* · *instance attribute*

```python
READ_ONLY = 'read_only'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1100-L1100)

### particula.execution.graph_capture.GraphCaptureFailureClassification.WRITER_MAY_HAVE_LAUNCHED

*attribute* · *class attribute* · *instance attribute*

```python
WRITER_MAY_HAVE_LAUNCHED = 'writer_may_have_launched'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1101-L1101)

## particula.execution.graph_capture.GraphCaptureLifecycle

*class* · *dataclass*

```python
class GraphCaptureLifecycle
```

Retain immutable host metadata for one graph-capture lifecycle.

This declaration records lifecycle intent only. It does not import Warp,
allocate or capture a native graph, replay work, or act on a resident
session.

**Attributes**

- `capability` (`GraphCaptureCapability`) — Exact graph-capture capability declaration retained by identity. Lifecycle creation requires it to be available.
- `signature` (`ResidentGraphCaptureSignature`) — Exact resident graph-capture identity signature.
- `state` (`GraphCaptureLifecycleState`) — Current lifecycle declaration.
- `first_invalidation_reason` (`GraphCaptureDriftReason | None`) — First structural drift reason, if recorded.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1104-L1179)

### particula.execution.graph_capture.GraphCaptureLifecycle.capability

*attribute* · *instance attribute*

```python
capability: GraphCaptureCapability
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1120-L1120)

### particula.execution.graph_capture.GraphCaptureLifecycle.first_invalidation_reason

*attribute* · *instance attribute*

```python
first_invalidation_reason: GraphCaptureDriftReason | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1123-L1123)

### particula.execution.graph_capture.GraphCaptureLifecycle.signature

*attribute* · *instance attribute*

```python
signature: ResidentGraphCaptureSignature
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1121-L1121)

### particula.execution.graph_capture.GraphCaptureLifecycle.state

*attribute* · *instance attribute*

```python
state: GraphCaptureLifecycleState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1122-L1122)

### particula.execution.graph_capture.GraphCaptureLifecycle.__init__

*method*

```python
def __init__(capability: GraphCaptureCapability, signature: ResidentGraphCaptureSignature, state: GraphCaptureLifecycleState, first_invalidation_reason: GraphCaptureDriftReason | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L0-L0)

### particula.execution.graph_capture.GraphCaptureLifecycle.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate lifecycle metadata types and reason-state invariants.

**Raises**

- `TypeError` — If a carrier, state, or invalidation reason has an
inexact type.
- `ValueError` — If the invalidation reason is inconsistent with the
lifecycle state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1125-L1179)

## particula.execution.graph_capture.GraphCaptureLifecycleState

*class*

```python
class GraphCaptureLifecycleState(str, Enum)
```

Bases: `str`, `Enum`

Enumerate host-only graph-capture lifecycle metadata states.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1086-L1094)

### particula.execution.graph_capture.GraphCaptureLifecycleState.CAPTURED

*attribute* · *class attribute* · *instance attribute*

```python
CAPTURED = 'captured'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1090-L1090)

### particula.execution.graph_capture.GraphCaptureLifecycleState.CLOSED

*attribute* · *class attribute* · *instance attribute*

```python
CLOSED = 'closed'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1094-L1094)

### particula.execution.graph_capture.GraphCaptureLifecycleState.FAULTED

*attribute* · *class attribute* · *instance attribute*

```python
FAULTED = 'faulted'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1092-L1092)

### particula.execution.graph_capture.GraphCaptureLifecycleState.INVALIDATED

*attribute* · *class attribute* · *instance attribute*

```python
INVALIDATED = 'invalidated'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1091-L1091)

### particula.execution.graph_capture.GraphCaptureLifecycleState.READY

*attribute* · *class attribute* · *instance attribute*

```python
READY = 'ready'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1089-L1089)

### particula.execution.graph_capture.GraphCaptureLifecycleState.RETIRED

*attribute* · *class attribute* · *instance attribute*

```python
RETIRED = 'retired'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1093-L1093)

## particula.execution.graph_capture.GraphCaptureNativeCallables

*class* · *dataclass*

```python
class GraphCaptureNativeCallables
```

Retain an adapter's native callable vocabulary by exact identity.

This record is vocabulary only: it holds no graph or executable handle.
capture invokes ``capture_begin()``, ``capture_end()``, and, after a
post-end
rejection, ``capture_release()``. A failure after begin invokes
``capture_abort()`` and releases a non-``None`` abort result.
It never invokes ``capture_instantiate`` or ``capture_launch``. In
particular, ``capture_release`` is a callable for a later owner to invoke,
not a retained cleanup action.

**Attributes**

- `capture_begin` (`Callable[..., object]`) — Callable that begins native graph capture.
- `capture_end` (`Callable[..., object]`) — Callable that ends native graph capture.
- `capture_instantiate` (`Callable[..., object]`) — Callable that instantiates a captured graph.
- `capture_launch` (`Callable[..., object]`) — Callable that launches an instantiated graph.
- `capture_release` (`Callable[..., object]`) — Callable that releases a native graph resource.
- `capture_abort` (`Callable[..., object]`) — Callable that aborts an incomplete native capture.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L145-L185)

### particula.execution.graph_capture.GraphCaptureNativeCallables.capture_abort

*attribute* · *class attribute* · *instance attribute*

```python
capture_abort: Callable[..., object] = _noop_capture_abort
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L172-L172)

### particula.execution.graph_capture.GraphCaptureNativeCallables.capture_begin

*attribute* · *instance attribute*

```python
capture_begin: Callable[..., object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L167-L167)

### particula.execution.graph_capture.GraphCaptureNativeCallables.capture_end

*attribute* · *instance attribute*

```python
capture_end: Callable[..., object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L168-L168)

### particula.execution.graph_capture.GraphCaptureNativeCallables.capture_instantiate

*attribute* · *instance attribute*

```python
capture_instantiate: Callable[..., object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L169-L169)

### particula.execution.graph_capture.GraphCaptureNativeCallables.capture_launch

*attribute* · *instance attribute*

```python
capture_launch: Callable[..., object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L170-L170)

### particula.execution.graph_capture.GraphCaptureNativeCallables.capture_release

*attribute* · *instance attribute*

```python
capture_release: Callable[..., object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L171-L171)

### particula.execution.graph_capture.GraphCaptureNativeCallables.__init__

*method*

```python
def __init__(capture_begin: Callable[..., object], capture_end: Callable[..., object], capture_instantiate: Callable[..., object], capture_launch: Callable[..., object], capture_release: Callable[..., object], capture_abort: Callable[..., object] = _noop_capture_abort) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L0-L0)

### particula.execution.graph_capture.GraphCaptureNativeCallables.__post_init__

*method*

```python
def __post_init__() -> None
```

Require every native vocabulary member to be callable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L174-L185)

## particula.execution.graph_capture.GraphCaptureRuntimeAdapter

*class*

```python
class GraphCaptureRuntimeAdapter(GraphCaptureRuntimeProbe, Protocol)
```

Bases: `particula.execution.graph_capture.GraphCaptureRuntimeProbe`, `Protocol`

Declare lazy runtime probes and callable resolution for qualification.

The adapter owns runtime-specific behavior and treats ``Device.native`` as
opaque. Qualification resolves runtime, device, and API availability in
order, then calls ``capture_callables`` once only after all checks pass. It
retains the returned exact vocabulary by identity and never invokes it.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L188-L206)

### particula.execution.graph_capture.GraphCaptureRuntimeAdapter.capture_callables

*method*

```python
def capture_callables(device: Device) -> GraphCaptureNativeCallables
```

Return native callable vocabulary for one exactly qualified device.

**Parameters**

- `device` (`Device`) — Exact device declaration whose capture API was qualified.

**Returns**

- (`GraphCaptureNativeCallables`) — Exact callable vocabulary retained for a later native capture
- (`GraphCaptureNativeCallables`) — phase; this call creates neither a native handle nor cleanup work.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L197-L206)

### particula.execution.graph_capture.GraphCaptureRuntimeAdapter.capture_api_available

*method*

```python
def capture_api_available(device: Device) -> bool
```

Inherited from: `particula.execution.graph_capture.GraphCaptureRuntimeProbe`

Return whether a declared device exposes a capture API.

**Parameters**

- `device` (`Device`) — Exact device declaration to assess.

**Returns**

- (`bool`) — ``True`` when the device exposes the required capture API.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L134-L142)

### particula.execution.graph_capture.GraphCaptureRuntimeAdapter.device_available

*method*

```python
def device_available(device: Device) -> bool
```

Inherited from: `particula.execution.graph_capture.GraphCaptureRuntimeProbe`

Return whether a declared device is available.

**Parameters**

- `device` (`Device`) — Exact device declaration to assess.

**Returns**

- (`bool`) — ``True`` when the declared device is available to the runtime.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L124-L132)

### particula.execution.graph_capture.GraphCaptureRuntimeAdapter.runtime_available

*method*

```python
def runtime_available() -> bool
```

Inherited from: `particula.execution.graph_capture.GraphCaptureRuntimeProbe`

Return whether the optional graph-capture runtime is available.

**Returns**

- (`bool`) — ``True`` when the runtime can be queried for graph-capture support.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L117-L122)

## particula.execution.graph_capture.GraphCaptureRuntimeProbe

*class*

```python
class GraphCaptureRuntimeProbe(Protocol)
```

Bases: `Protocol`

Declare caller-owned lazy runtime checks for graph-capture support.

Implementations must return literal ``bool`` values. The resolver invokes
methods only after the preceding availability condition succeeds.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L110-L142)

### particula.execution.graph_capture.GraphCaptureRuntimeProbe.capture_api_available

*method*

```python
def capture_api_available(device: Device) -> bool
```

Return whether a declared device exposes a capture API.

**Parameters**

- `device` (`Device`) — Exact device declaration to assess.

**Returns**

- (`bool`) — ``True`` when the device exposes the required capture API.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L134-L142)

### particula.execution.graph_capture.GraphCaptureRuntimeProbe.device_available

*method*

```python
def device_available(device: Device) -> bool
```

Return whether a declared device is available.

**Parameters**

- `device` (`Device`) — Exact device declaration to assess.

**Returns**

- (`bool`) — ``True`` when the declared device is available to the runtime.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L124-L132)

### particula.execution.graph_capture.GraphCaptureRuntimeProbe.runtime_available

*method*

```python
def runtime_available() -> bool
```

Return whether the optional graph-capture runtime is available.

**Returns**

- (`bool`) — ``True`` when the runtime can be queried for graph-capture support.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L117-L122)

## particula.execution.graph_capture.PreparedGraphCaptureQualification

*class* · *dataclass*

```python
class PreparedGraphCaptureQualification
```

Retain one exact READY qualification without a native handle or cleanup.

Qualification retains the prepared resident binding, published capture
resources, and
native callable vocabulary by identity; it copies no payloads. It creates
no graph or executable handle, owns no cleanup callback, and does not
invoke any retained callable. Successful qualification leaves the binding's
lifecycle in READY; capture/replay alone own native capture, handles,
release, and
cleanup.

**Attributes**

- `binding` (`ResidentGraphCaptureBinding`) — Attached resident graph-capture binding being qualified.
- `lifecycle` (`GraphCaptureLifecycle`) — READY lifecycle retained by the binding.
- `signature` (`ResidentGraphCaptureSignature`) — Structural identity signature for the request.
- `request` (`object`) — Exact resident simulation request.
- `session` (`object`) — Exact resident session attached to the request.
- `registry` (`object`) — Exact resource registry pinned to the session.
- `guard` (`object`) — Exact closed step guard attached to the session.
- `prepared` (`object`) — Exact prepared resident simulation.
- `timestep` (`object`) — Exact prepared timestep metadata.
- `capture_requirements` (`object`) — Published capture-resource requirements.
- `capture_set` (`object`) — Published capture-resource set.
- `capture_report` (`object`) — Cached logical-byte report for the capture set.
- `device` (`Device`) — Exact non-CPU Warp device qualified for capture.
- `dimensions` (`object`) — Resident dimensions retained by the session.
- `graph` (`object`) — Prepared graph declaration.
- `schedule` (`object`) — Prepared schedule declaration.
- `ordered_node_ids` (`tuple[object, ...]`) — Ordered schedule-node identifiers.
- `duration` (`object`) — Prepared timestep duration.
- `duration_is_identity` (`bool`) — Whether prepared and timestep durations are the same object rather than merely equal values.
- `primary_arrays` (`tuple[object, ...]`) — Identity tuple of resident primary arrays.
- `resource_views` (`tuple[object, ...]`) — Identity tuple of resident resource views.
- `native_callables` (`GraphCaptureNativeCallables`) — Adapter-provided callable vocabulary for later use.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1645-L1705)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.binding

*attribute* · *instance attribute*

```python
binding: ResidentGraphCaptureBinding
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1684-L1684)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.capture_report

*attribute* · *instance attribute*

```python
capture_report: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1695-L1695)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.capture_requirements

*attribute* · *instance attribute*

```python
capture_requirements: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1693-L1693)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.capture_set

*attribute* · *instance attribute*

```python
capture_set: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1694-L1694)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.device

*attribute* · *instance attribute*

```python
device: Device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1696-L1696)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.dimensions

*attribute* · *instance attribute*

```python
dimensions: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1697-L1697)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.duration

*attribute* · *instance attribute*

```python
duration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1701-L1701)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.duration_is_identity

*attribute* · *instance attribute*

```python
duration_is_identity: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1702-L1702)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.graph

*attribute* · *instance attribute*

```python
graph: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1698-L1698)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.guard

*attribute* · *instance attribute*

```python
guard: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1690-L1690)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.lifecycle

*attribute* · *instance attribute*

```python
lifecycle: GraphCaptureLifecycle
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1685-L1685)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.native_callables

*attribute* · *instance attribute*

```python
native_callables: GraphCaptureNativeCallables
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1705-L1705)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.ordered_node_ids

*attribute* · *instance attribute*

```python
ordered_node_ids: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1700-L1700)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.prepared

*attribute* · *instance attribute*

```python
prepared: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1691-L1691)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.primary_arrays

*attribute* · *instance attribute*

```python
primary_arrays: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1703-L1703)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.registry

*attribute* · *instance attribute*

```python
registry: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1689-L1689)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.request

*attribute* · *instance attribute*

```python
request: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1687-L1687)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.resource_views

*attribute* · *instance attribute*

```python
resource_views: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1704-L1704)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.schedule

*attribute* · *instance attribute*

```python
schedule: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1699-L1699)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.session

*attribute* · *instance attribute*

```python
session: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1688-L1688)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.signature

*attribute* · *instance attribute*

```python
signature: ResidentGraphCaptureSignature
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1686-L1686)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.timestep

*attribute* · *instance attribute*

```python
timestep: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1692-L1692)

### particula.execution.graph_capture.PreparedGraphCaptureQualification.__init__

*method*

```python
def __init__(binding: ResidentGraphCaptureBinding, lifecycle: GraphCaptureLifecycle, signature: ResidentGraphCaptureSignature, request: object, session: object, registry: object, guard: object, prepared: object, timestep: object, capture_requirements: object, capture_set: object, capture_report: object, device: Device, dimensions: object, graph: object, schedule: object, ordered_node_ids: tuple[object, ...], duration: object, duration_is_identity: bool, primary_arrays: tuple[object, ...], resource_views: tuple[object, ...], native_callables: GraphCaptureNativeCallables) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L0-L0)

## particula.execution.graph_capture.ResidentGraphCaptureBinding

*class* · *dataclass*

```python
class ResidentGraphCaptureBinding
```

Retain one exact resident binding and its mutable lifecycle metadata.

This direct-module-only carrier owns lifecycle successors for the attached
final request. It is metadata only: it neither captures nor replays a graph,
accesses resident payloads, transfers data, synchronizes, or falls back.

**Attributes**

- `lifecycle` (`GraphCaptureLifecycle`) — Current immutable lifecycle metadata retained by the binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1442-L1473)

### particula.execution.graph_capture.ResidentGraphCaptureBinding._guard

*attribute* · *instance attribute*

```python
_guard: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1457-L1457)

### particula.execution.graph_capture.ResidentGraphCaptureBinding._lifecycle

*attribute* · *instance attribute*

```python
_lifecycle: GraphCaptureLifecycle
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1458-L1458)

### particula.execution.graph_capture.ResidentGraphCaptureBinding._registry

*attribute* · *instance attribute*

```python
_registry: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1456-L1456)

### particula.execution.graph_capture.ResidentGraphCaptureBinding._request

*attribute* · *instance attribute*

```python
_request: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1454-L1454)

### particula.execution.graph_capture.ResidentGraphCaptureBinding._session

*attribute* · *instance attribute*

```python
_session: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1455-L1455)

### particula.execution.graph_capture.ResidentGraphCaptureBinding.lifecycle

*property*

```python
lifecycle: GraphCaptureLifecycle
```

Return the lifecycle currently owned by this binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1471-L1473)

### particula.execution.graph_capture.ResidentGraphCaptureBinding.__init__

*method*

```python
def __init__(_request: object, _session: object, _registry: object, _guard: object, _lifecycle: GraphCaptureLifecycle) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L0-L0)

### particula.execution.graph_capture.ResidentGraphCaptureBinding.__post_init__

*method*

```python
def __post_init__() -> None
```

Require the exact, mutually identity-bound resident carriers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1460-L1468)

## particula.execution.graph_capture.ResidentGraphCaptureSignature

*class* · *dataclass*

```python
class ResidentGraphCaptureSignature
```

Retain immutable identity metadata for one resident request.

Every tuple holds references rather than payload values. The names mirror
the ordered drift groups used by
:func:`compare_resident_graph_capture_signature`.

**Attributes**

- `request` (`object`) — Exact resident request instance.
- `session` (`object`) — Request-bound resident session.
- `device` (`object`) — Session metadata device declaration.
- `dimensions` (`object`) — Session resident dimensions.
- `primary_containers` (`tuple[object, ...]`) — Particle, gas, and environment containers.
- `primary_arrays` (`tuple[object, ...]`) — Primary container arrays in schema order.
- `resource_views` (`tuple[object, ...]`) — Request-owned published process and communication views.
- `graph` (`tuple[object, ...]`) — Graph and its declaration tuples.
- `schedule` (`tuple[object, ...]`) — Schedule and its declaration tuples.
- `schedule_order` (`tuple[object, ...]`) — Resolved ordered schedule-node identifiers.
- `diagnostics` (`tuple[object, ...]`) — Diagnostic plan, registrations, and output identities.
- `communication` (`tuple[object, ...]`) — Optional communication declaration and view identities.
- `configurations` (`tuple[object, ...]`) — Request-bound process configurations followed by the exact capture requirements, published resource set, and cached logical-byte report identities.
- `rng_resources` (`tuple[object, ...]`) — Coagulation and wall-loss RNG array identities.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L330-L391)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.communication

*attribute* · *instance attribute*

```python
communication: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L368-L368)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.configurations

*attribute* · *instance attribute*

```python
configurations: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L369-L369)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.device

*attribute* · *instance attribute*

```python
device: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L359-L359)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.diagnostics

*attribute* · *instance attribute*

```python
diagnostics: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L367-L367)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.dimensions

*attribute* · *instance attribute*

```python
dimensions: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L360-L360)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.graph

*attribute* · *instance attribute*

```python
graph: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L364-L364)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.primary_arrays

*attribute* · *instance attribute*

```python
primary_arrays: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L362-L362)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.primary_containers

*attribute* · *instance attribute*

```python
primary_containers: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L361-L361)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.request

*attribute* · *instance attribute*

```python
request: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L357-L357)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.resource_views

*attribute* · *instance attribute*

```python
resource_views: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L363-L363)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.rng_resources

*attribute* · *instance attribute*

```python
rng_resources: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L370-L370)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.schedule

*attribute* · *instance attribute*

```python
schedule: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L365-L365)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.schedule_order

*attribute* · *instance attribute*

```python
schedule_order: tuple[object, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L366-L366)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.session

*attribute* · *instance attribute*

```python
session: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L358-L358)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.__init__

*method*

```python
def __init__(request: object, session: object, device: object, dimensions: object, primary_containers: tuple[object, ...], primary_arrays: tuple[object, ...], resource_views: tuple[object, ...], graph: tuple[object, ...], schedule: tuple[object, ...], schedule_order: tuple[object, ...], diagnostics: tuple[object, ...], communication: tuple[object, ...], configurations: tuple[object, ...], rng_resources: tuple[object, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L0-L0)

### particula.execution.graph_capture.ResidentGraphCaptureSignature.__post_init__

*method*

```python
def __post_init__() -> None
```

Require exact immutable tuples for grouped identity metadata.

**Raises**

- `TypeError` — If a grouped metadata field is not an exact tuple.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L372-L391)

## particula.execution.graph_capture._DuplicateNativeHandleError

*class*

```python
class _DuplicateNativeHandleError(ValueError)
```

Bases: `ValueError`

Reject reuse of an opaque native handle identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1792-L1793)

## particula.execution.graph_capture._NativeGraphRecord

*class* · *dataclass*

```python
class _NativeGraphRecord
```

Own one opaque handle and its launch/release provenance privately.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1796-L1804)

### particula.execution.graph_capture._NativeGraphRecord.captured

*attribute* · *instance attribute*

```python
captured: CapturedResidentGraph
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1800-L1800)

### particula.execution.graph_capture._NativeGraphRecord.handle

*attribute* · *instance attribute*

```python
handle: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1801-L1801)

### particula.execution.graph_capture._NativeGraphRecord.launch_leases

*attribute* · *class attribute* · *instance attribute*

```python
launch_leases: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1804-L1804)

### particula.execution.graph_capture._NativeGraphRecord.release

*attribute* · *instance attribute*

```python
release: Callable[..., object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1802-L1802)

### particula.execution.graph_capture._NativeGraphRecord.state

*attribute* · *class attribute* · *instance attribute*

```python
state: _NativeGraphState = _NativeGraphState.LIVE
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1803-L1803)

### particula.execution.graph_capture._NativeGraphRecord.__init__

*method*

```python
def __init__(captured: CapturedResidentGraph, handle: object, release: Callable[..., object], state: _NativeGraphState = _NativeGraphState.LIVE, launch_leases: int = 0) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L0-L0)

## particula.execution.graph_capture._NativeGraphState

*class*

```python
class _NativeGraphState(str, Enum)
```

Bases: `str`, `Enum`

Track private native graph provenance through one-way release states.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1784-L1789)

### particula.execution.graph_capture._NativeGraphState.LIVE

*attribute* · *class attribute* · *instance attribute*

```python
LIVE = 'live'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1787-L1787)

### particula.execution.graph_capture._NativeGraphState.RELEASED

*attribute* · *class attribute* · *instance attribute*

```python
RELEASED = 'released'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1789-L1789)

### particula.execution.graph_capture._NativeGraphState.RELEASING

*attribute* · *class attribute* · *instance attribute*

```python
RELEASING = 'releasing'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1788-L1788)

## particula.execution.graph_capture._acquire_captured_graph_launch

*function*

```python
def _acquire_captured_graph_launch(captured: object) -> tuple[CapturedResidentGraph, _NativeGraphRecord]
```

Authenticate a live record and acquire its per-binding launch lease.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2010-L2023)

## particula.execution.graph_capture._attach_resident_graph_capture_binding

*function*

```python
def _attach_resident_graph_capture_binding(request: object, binding: object) -> None
```

Attach one exact binding to a final frozen request exactly once.

The request must be constructed without an attachment first. After all
retained identities are checked, this construction-only helper performs the
sole assignment to the frozen request's optional binding field and records
the binding privately on its exact session. Exact reattachment is a no-op;
a different binding for the same session is rejected.

**Parameters**

- `request` (`object`) — Exact resident simulation request to attach.
- `binding` (`object`) — Exact binding retaining the same request and resident carriers.

**Raises**

- `TypeError` — If either argument is not the required exact concrete type.
- `ValueError` — If the binding retains another request, the request is
already attached, or the session retains a different binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1566-L1620)

## particula.execution.graph_capture._capture_resource_requirements_type

*function*

```python
def _capture_resource_requirements_type() -> type[object]
```

Lazily return exact capture-resource requirements carrier type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L438-L442)

## particula.execution.graph_capture._capture_resource_set_type

*function*

```python
def _capture_resource_set_type() -> type[object]
```

Lazily return exact published capture-resource set carrier type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L445-L449)

## particula.execution.graph_capture._classify_capture_operational_failure

*function*

```python
def _classify_capture_operational_failure(qualification: PreparedGraphCaptureQualification) -> BaseException | None
```

Classify a capture failure when its exact binding remains attached.

**Parameters**

- `qualification` (`PreparedGraphCaptureQualification`) — Exact qualification associated with the failed capture.

**Returns**

- (`BaseException | None`) — The classification error when classification fails, otherwise ``None``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2364-L2399)

## particula.execution.graph_capture._compare_cached_admission_signature

*function*

```python
def _compare_cached_admission_signature(signature: ResidentGraphCaptureSignature, request: object) -> GraphCaptureCompatibility
```

Compare fixed admission identities without rebuilding a signature.

The registry and publication gates separately validate pinned arrays and
capture resources. This fast path checks the request-owned structural
carriers that supported preparation retains by identity. The exhaustive
public comparison remains the drift-diagnostic fallback.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L774-L1015)

## particula.execution.graph_capture._fault_resident_graph_capture_after_classification_failure

*function*

```python
def _fault_resident_graph_capture_after_classification_failure(binding: object) -> None
```

Fault an attached binding after classification itself fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2838-L2852)

## particula.execution.graph_capture._identity_tuple

*function*

```python
def _identity_tuple(*values: object) -> tuple[object, ...]
```

Build a tuple whose members are compared by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L452-L454)

## particula.execution.graph_capture._lifecycle_successor

*function*

```python
def _lifecycle_successor(lifecycle: GraphCaptureLifecycle, state: GraphCaptureLifecycleState, reason: GraphCaptureDriftReason | None) -> GraphCaptureLifecycle
```

Create a successor retaining qualification capability and signature
identities.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1189-L1202)

## particula.execution.graph_capture._nested_sidecar_identities

*function*

```python
def _nested_sidecar_identities(value: object) -> tuple[object, ...]
```

Flatten dataclass sidecar leaves in deterministic declaration order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L457-L465)

## particula.execution.graph_capture._noop_capture_abort

*function*

```python
def _noop_capture_abort() -> None
```

Provide a compatibility abort for adapters without native cleanup.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L46-L47)

## particula.execution.graph_capture._notify_resident_graph_capture

*function*

```python
def _notify_resident_graph_capture(session: object, registry: object, guard: object, cause: str) -> None
```

Route an exact resident lifecycle event to an attached graph binding.

Unattached sessions are successful no-ops. An attached binding must retain
the exact session, registry, and closed guard; the graph-capture owner then
makes its issued records stale and performs the requested teardown.

**Parameters**

- `session` (`object`) — Resident session reporting the event.
- `registry` (`object`) — Registry that must be pinned to ``session``.
- `guard` (`object`) — Closed guard that must be bound to ``session`` and ``registry``.
- `cause` (`str`) — Teardown event forwarded to the graph-capture owner.

**Raises**

- `TypeError` — If an attached object is not an exact graph-capture binding.
- `ValueError` — If an attached binding does not match the exact context.
- `BaseException` — Propagates a graph lifecycle or native-release failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1927-L1963)

## particula.execution.graph_capture._prepared_resident_simulation_type

*function*

```python
def _prepared_resident_simulation_type() -> type[object]
```

Lazily return the exact prepared resident simulation carrier type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L422-L428)

## particula.execution.graph_capture._prepared_resident_timestep_type

*function*

```python
def _prepared_resident_timestep_type() -> type[object]
```

Lazily return the exact prepared resident timestep carrier type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L431-L435)

## particula.execution.graph_capture._raise_capture_operational_failure

*function*

```python
def _raise_capture_operational_failure(qualification: PreparedGraphCaptureQualification, error: BaseException, cleanup_error: BaseException | None = None) -> NoReturn
```

Raise an operational capture error while retaining its primary cause.

**Parameters**

- `qualification` (`PreparedGraphCaptureQualification`) — Exact qualification associated with the failed capture.
- `error` (`BaseException`) — Primary operational error to propagate.
- `cleanup_error` (`BaseException | None`) (default: `None`) — Optional capture-end or release error to chain.

**Raises**

- `BaseException` — Always raises ``error`` with any cleanup or
classification failure chained as its cause.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2402-L2425)

## particula.execution.graph_capture._raise_replay_operational_failure

*function*

```python
def _raise_replay_operational_failure(qualification: PreparedGraphCaptureQualification, token: object, error: BaseException) -> NoReturn
```

Handle a writer-capable replay failure while preserving ``error``.

This helper runs only after a resident token opens and either native launch
or token completion fails. It invokes existing resident writer-capable
cleanup, then faults capture lifecycle metadata through classification. It
does not retry, fall back, recapture, release a handle, or promise rollback.

**Parameters**

- `qualification` (`PreparedGraphCaptureQualification`) — Exact qualification associated with the failed replay.
- `token` (`object`) — Open resident-step token to clean up.
- `error` (`BaseException`) — Primary launch or completion error to propagate.

**Raises**

- `BaseException` — Always raises ``error`` with cleanup or classification
failure chained as its cause.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2634-L2674)

## particula.execution.graph_capture._raise_unavailable_qualification

*function*

```python
def _raise_unavailable_qualification(capability: GraphCaptureCapability) -> None
```

Map a resolved unavailable capability to a deterministic error.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2045-L2067)

## particula.execution.graph_capture._register_captured_graph

*function*

```python
def _register_captured_graph(captured: CapturedResidentGraph, handle: object) -> None
```

Publish exactly one live provenance record for an opaque identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1991-L2007)

## particula.execution.graph_capture._registry_type

*function*

```python
def _registry_type() -> type[object]
```

Lazily return the exact pinned-resource-registry carrier type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L415-L419)

## particula.execution.graph_capture._release_captured_graph_launch

*function*

```python
def _release_captured_graph_launch(record: _NativeGraphRecord) -> None
```

Release one exact launch lease and wake pending teardown.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2026-L2032)

## particula.execution.graph_capture._release_issued_captured_graphs_for_binding

*function*

```python
def _release_issued_captured_graphs_for_binding(binding: ResidentGraphCaptureBinding) -> None
```

Release and unregister capture-issued handles owned by one binding.

Removing each record before its native release makes the operation
exactly-once even when native cleanup fails.  This capture lifecycle helper
is
deliberately not used by replay, which only authenticates and launches
an issued handle.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1811-L1846)

## particula.execution.graph_capture._require_adapter_method

*function*

```python
def _require_adapter_method(adapter: object, name: str) -> Callable[..., object]
```

Return one callable adapter method without invoking it.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2035-L2042)

## particula.execution.graph_capture._require_attached_resident_binding

*function*

```python
def _require_attached_resident_binding(binding: object) -> ResidentGraphCaptureBinding
```

Require an exact binding attached to its retained final request.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1548-L1563)

## particula.execution.graph_capture._require_binding

*function*

```python
def _require_binding(binding: object) -> ResidentGraphCaptureBinding
```

Require an exact graph-capture binding before reading its fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1541-L1545)

## particula.execution.graph_capture._require_bool

*function*

```python
def _require_bool(result: object, name: str) -> bool
```

Require a literal boolean result from a probe.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L217-L221)

## particula.execution.graph_capture._require_exact_resident_carrier

*function*

```python
def _require_exact_resident_carrier(value: object, name: str, expected_name: str, resolver: Callable[[], type[object]]) -> object
```

Reject obvious inexact input before lazily importing resident modules.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1476-L1487)

## particula.execution.graph_capture._require_issued_captured_graph

*function*

```python
def _require_issued_captured_graph(captured: object) -> CapturedResidentGraph
```

Require an authentic capture-issued record and its exact opaque handle.

This provenance lookup is metadata-only. It does not inspect the opaque
handle, allocate a resource, transfer data, read back, synchronize, or
probe native state.

**Parameters**

- `captured` (`object`) — Candidate captured-record object.

**Returns**

- (`CapturedResidentGraph`) — The exact registered captured graph.

**Raises**

- `TypeError` — If ``captured`` is not an exact ``CapturedResidentGraph``.
- `ValueError` — If the record is unissued or its handle identity changed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1966-L1988)

## particula.execution.graph_capture._require_lifecycle

*function*

```python
def _require_lifecycle(lifecycle: object) -> GraphCaptureLifecycle
```

Require an exact lifecycle record before accessing its metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1182-L1186)

## particula.execution.graph_capture._require_probe_method

*function*

```python
def _require_probe_method(probe: object, name: str) -> Callable[..., object]
```

Return one callable probe method or raise a deterministic error.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L209-L214)

## particula.execution.graph_capture._resident_guard_type

*function*

```python
def _resident_guard_type() -> type[object]
```

Lazily return the exact resident-step-guard carrier type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L408-L412)

## particula.execution.graph_capture._resident_request_type

*function*

```python
def _resident_request_type() -> type[object]
```

Lazily import the exact concrete request type after rejection checks.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L394-L398)

## particula.execution.graph_capture._resident_session_type

*function*

```python
def _resident_session_type() -> type[object]
```

Lazily return the exact resident-session carrier type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L401-L405)

## particula.execution.graph_capture._same_identity

*function*

```python
def _same_identity(left: object, right: object) -> bool
```

Compare arbitrary scalar or tuple signature entries by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L765-L771)

## particula.execution.graph_capture._teardown_resident_graph_capture

*function*

```python
def _teardown_resident_graph_capture(binding: ResidentGraphCaptureBinding, cause: str, compatibility: GraphCaptureCompatibility | None = None) -> None
```

Make issued handles stale, transition lifecycle, then release once.

**Parameters**

- `binding` (`ResidentGraphCaptureBinding`) — Exact attached binding that owns the issued records.
- `cause` (`str`) — Lifecycle event that requires invalidation, faulting, retirement, or closure.
- `compatibility` (`GraphCaptureCompatibility | None`) (default: `None`) — Ordered drift result required only for structural drift.

**Raises**

- `TypeError` — If ``binding`` is invalid or structural drift lacks an exact
compatibility result.
- `ValueError` — If ``cause`` is unknown or cannot transition the lifecycle.
- `BaseException` — Propagates a native handle-release failure after records
are unregistered and lifecycle metadata is nondispatchable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1849-L1924)

## particula.execution.graph_capture._validate_prepared_graph_capture_qualification

*function*

```python
def _validate_prepared_graph_capture_qualification(qualification: object) -> PreparedGraphCaptureQualification
```

Recheck one READY qualification without invoking native callables.

**Parameters**

- `qualification` (`object`) — Candidate exact qualification retaining the prepared binding and native callable vocabulary.

**Returns**

- (`PreparedGraphCaptureQualification`) — The same validated qualification object.

**Raises**

- `TypeError` — If ``qualification`` or a retained concrete carrier has an
invalid exact type.
- `ValueError` — If the READY binding, resident session, signature, prepared
operations, or published capture resource set has changed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2280-L2361)

## particula.execution.graph_capture._validate_replay_captured_graph

*function*

```python
def _validate_replay_captured_graph(captured: object, duration: object) -> PreparedGraphCaptureQualification
```

Validate an authentic replay record without inspecting payloads.

Provenance and retained links are checked before the captured-binding gate,
duration validation, token entry, or native launch. The gate preserves its
canonical first-drift ordering. Pinned array values and resident RNG words
may change; their retained object identities may not.

**Parameters**

- `captured` (`object`) — Candidate capture-issued captured graph.
- `duration` (`object`) — Requested replay duration.

**Returns**

- (`PreparedGraphCaptureQualification`) — The exact qualification retained by the authenticated record.

**Raises**

- `TypeError` — If the record or retained carrier has an invalid exact type,
or ``duration`` is not a valid resident-step duration.
- `ValueError` — If provenance, metadata, binding admission, device, or
duration compatibility fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2536-L2631)

## particula.execution.graph_capture._validate_resident_binding

*function*

```python
def _validate_resident_binding(request: object, session: object, registry: object, guard: object, lifecycle: object) -> None
```

Validate retained identity links without reading resident payloads.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1490-L1538)

## particula.execution.graph_capture.capture_prepared_resident_graph

*function*

```python
def capture_prepared_resident_graph(qualification: object) -> CapturedResidentGraph
```

Capture one qualified prepared timestep and retain its opaque handle.

Only native ``capture_begin()``, retained prepared dispatch, and
``capture_end()`` run inside the capture window. A literal ``None`` end
result is rejected. Only a successful capture that publishes CAPTURED
metadata issues a replay-authentic record and registers its opaque handle
by identity. A successful end handle is released exactly once only if
post-end validation or lifecycle completion rejects it. This concrete-only
operation neither instantiates, launches, nor replays a graph, and it
performs no token, allocation, transfer, readback, or synchronization work
in the capture window.

**Parameters**

- `qualification` (`object`) — Exact READY qualification returned by :func:`qualify_prepared_resident_graph_capture`.

**Returns**

- (`CapturedResidentGraph`) — Immutable record retaining the captured lifecycle successor and exact
- (`CapturedResidentGraph`) — binding metadata. The native ``capture_end()`` handle remains in
- (`CapturedResidentGraph`) — private module-owned provenance.

**Raises**

- `TypeError` — If ``qualification`` is not an exact valid qualification.
- `ValueError` — If retained READY metadata has drifted, capture is already
active for the binding, or lifecycle completion cannot publish a
CAPTURED successor.
- `BaseException` — Propagates native begin, dispatch, end, or release errors
while preserving the operational error as the primary cause.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2428-L2533)

## particula.execution.graph_capture.classify_graph_capture_failure

*function*

```python
def classify_graph_capture_failure(lifecycle: GraphCaptureLifecycle, classification: GraphCaptureFailureClassification) -> GraphCaptureLifecycle
```

Classify host failure metadata without executing a resident operation.

This declaration-only operation does not import Warp or act on a resident
session.

**Parameters**

- `lifecycle` (`GraphCaptureLifecycle`) — Lifecycle metadata associated with the failure.
- `classification` (`GraphCaptureFailureClassification`) — Whether a writer may have launched before failure.

**Returns**

- (`GraphCaptureLifecycle`) — The original record for read-only or faulted outcomes, or a faulted
- (`GraphCaptureLifecycle`) — successor when a writer may have launched.

**Raises**

- `TypeError` — If either argument has an inexact type.
- `ValueError` — If the lifecycle is retired or closed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1309-L1351)

## particula.execution.graph_capture.classify_resident_graph_capture_writer_failure

*function*

```python
def classify_resident_graph_capture_writer_failure(binding: object) -> None
```

Record a scheduler-confirmed possible writer failure on a binding.

The scheduler calls this only after its existing cleanup determines that a
writer may have launched. It makes issued handles stale before their one
native release; it does not retry, roll back, or act on resident payloads.

**Parameters**

- `binding` (`object`) — Exact binding whose lifecycle receives the classification.

**Raises**

- `TypeError` — If ``binding`` is not an exact binding.
- `ValueError` — If its lifecycle is retired or closed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2813-L2835)

## particula.execution.graph_capture.close_graph_capture

*function*

```python
def close_graph_capture(lifecycle: GraphCaptureLifecycle) -> GraphCaptureLifecycle
```

Close host metadata without releasing native resources or a session.

**Parameters**

- `lifecycle` (`GraphCaptureLifecycle`) — Lifecycle metadata to close.

**Returns**

- (`GraphCaptureLifecycle`) — A closed successor, or the original record when already closed.

**Raises**

- `TypeError` — If ``lifecycle`` is not an exact lifecycle record.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1418-L1439)

## particula.execution.graph_capture.close_resident_graph_capture

*function*

```python
def close_resident_graph_capture(binding: object) -> GraphCaptureLifecycle
```

Close lifecycle metadata owned by one exact attached binding.

Repeated closure returns the identical closed lifecycle. This operation does
not dispatch or mutate resident payloads. It makes any capture-issued record
stale and releases its native handle exactly once before terminal closure.

**Parameters**

- `binding` (`object`) — Exact attached resident graph-capture binding to close.

**Returns**

- (`GraphCaptureLifecycle`) — Closed lifecycle metadata now owned by ``binding``.

**Raises**

- `TypeError` — If ``binding`` is not an exact binding.
- `ValueError` — If retained ownership or attachment identities are stale.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2915-L2934)

## particula.execution.graph_capture.compare_resident_graph_capture_signature

*function*

```python
def compare_resident_graph_capture_signature(signature: ResidentGraphCaptureSignature, request: object, *, admission_token: ResidentGraphCaptureSignature | None = None) -> GraphCaptureCompatibility
```

Compare a request with a signature and return its first identity drift.

By default, a fresh metadata-only signature is created for complete ordered
drift diagnosis. Admission callers that already validated their exact
frozen binding may pass that retained signature itself as
``admission_token``. The identity token provides the compatible fast path
without rebuilding signature tuples; any different token fails closed.

**Parameters**

- `signature` (`ResidentGraphCaptureSignature`) — Exact baseline signature to compare.
- `request` (`object`) — Exact ``ResidentSimulationRequest`` to assess.
- `admission_token` (`ResidentGraphCaptureSignature | None`) (default: `None`) — Optional exact retained signature authorizing the cached compatible-admission path.

**Returns**

- (`GraphCaptureCompatibility`) — Compatibility result containing the first changed group, if any.

**Raises**

- `TypeError` — If ``signature`` is inexact or ``request`` is not an exact
``ResidentSimulationRequest``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1018-L1083)

## particula.execution.graph_capture.complete_graph_capture

*function*

```python
def complete_graph_capture(lifecycle: GraphCaptureLifecycle) -> GraphCaptureLifecycle
```

Declare capture completion without native or resident-session work.

**Parameters**

- `lifecycle` (`GraphCaptureLifecycle`) — Ready lifecycle metadata to transition.

**Returns**

- (`GraphCaptureLifecycle`) — A captured successor retaining the capability and signature identities.

**Raises**

- `TypeError` — If ``lifecycle`` is not an exact lifecycle record.
- `ValueError` — If the lifecycle is not ready.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1241-L1263)

## particula.execution.graph_capture.complete_resident_graph_capture

*function*

```python
def complete_resident_graph_capture(binding: object) -> GraphCaptureLifecycle
```

Explicitly declare capture completion on one retained binding.

This metadata transition does not perform native graph capture or resident
work.

**Parameters**

- `binding` (`object`) — Exact resident graph-capture binding to transition.

**Returns**

- (`GraphCaptureLifecycle`) — Captured lifecycle metadata now owned by ``binding``.

**Raises**

- `TypeError` — If ``binding`` is not an exact binding.
- `ValueError` — If the retained carriers are inconsistent or the lifecycle
is not ready.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1623-L1642)

## particula.execution.graph_capture.create_graph_capture_lifecycle

*function*

```python
def create_graph_capture_lifecycle(capability: GraphCaptureCapability, signature: ResidentGraphCaptureSignature) -> GraphCaptureLifecycle
```

Create ready host metadata without capturing or acting on a session.

This declaration-only operation does not import Warp, allocate a graph, or
validate or mutate a resident binding.

**Parameters**

- `capability` (`GraphCaptureCapability`) — Available graph-capture capability metadata.
- `signature` (`ResidentGraphCaptureSignature`) — Identity signature for the resident request.

**Returns**

- (`GraphCaptureLifecycle`) — A new ready lifecycle record retaining both input carriers by identity.

**Raises**

- `TypeError` — If either qualification carrier is inexact.
- `ValueError` — If graph capture is not available.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1205-L1238)

## particula.execution.graph_capture.create_resident_graph_capture_signature

*function*

```python
def create_resident_graph_capture_signature(request: object, *, validate_capture_resources: bool = True, cached_capture_set: object | None = None) -> ResidentGraphCaptureSignature
```

Create an identity-only structural signature for an exact request.

The concrete request type is imported lazily after early inexact-request
rejection. This function retains existing request metadata and published
views only; it does not inspect array payloads or acquire resources.

**Parameters**

- `request` (`object`) — Exact ``ResidentSimulationRequest`` to describe.
- `validate_capture_resources` (`bool`) (default: `True`) — Whether to validate the published resource set before creating the signature.
- `cached_capture_set` (`object | None`) (default: `None`) — Previously validated resource set used when ``validate_capture_resources`` is ``False``.

**Returns**

- (`ResidentGraphCaptureSignature`) — Immutable identity metadata grouped in comparison precedence order.

**Raises**

- `TypeError` — If ``request`` is not an exact
``ResidentSimulationRequest``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L537-L762)

## particula.execution.graph_capture.gate_resident_graph_capture

*function*

```python
def gate_resident_graph_capture(binding: object) -> None
```

Fail closed unless a captured binding remains exactly dispatchable.

The gate verifies the attached request, resident binding, closed guard,
active pinned session, available CUDA capability, captured lifecycle, and
unchanged structural signature before scheduler token entry. Structural
drift invalidates a captured lifecycle; all other rejection is read-only.
This operation neither captures nor replays graphs, transfers data,
synchronizes, acquires resources, or falls back.

**Parameters**

- `binding` (`object`) — Exact resident graph-capture binding to admit.

**Raises**

- `TypeError` — If ``binding`` or its retained carriers have inexact types.
- `ValueError` — If the binding is stale, unavailable, not captured, or
structurally incompatible with its resident request.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2741-L2810)

## particula.execution.graph_capture.invalidate_graph_capture

*function*

```python
def invalidate_graph_capture(lifecycle: GraphCaptureLifecycle, compatibility: GraphCaptureCompatibility) -> GraphCaptureLifecycle
```

Record structural invalidation host metadata without comparing requests.

This operation neither imports Warp nor acts on a resident session.

**Parameters**

- `lifecycle` (`GraphCaptureLifecycle`) — Captured or already-invalidated lifecycle metadata.
- `compatibility` (`GraphCaptureCompatibility`) — qualification compatibility result containing the drift outcome.

**Returns**

- (`GraphCaptureLifecycle`) — The original record for compatible or repeated invalidation, otherwise
- (`GraphCaptureLifecycle`) — an invalidated successor retaining the first drift reason.

**Raises**

- `TypeError` — If either argument has an inexact type.
- `ValueError` — If the lifecycle is not captured or invalidated.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1266-L1306)

## particula.execution.graph_capture.qualify_prepared_resident_graph_capture

*function*

```python
def qualify_prepared_resident_graph_capture(binding: object, prepared: object, capture_set: object, adapter: GraphCaptureRuntimeAdapter) -> PreparedGraphCaptureQualification
```

Qualify one exact prepared READY binding for later native capture.

This metadata-only qualification boundary validates existing exact
identities, then
lazily resolves the adapter vocabulary in runtime, device, API, and
callable order. It returns a fresh frozen record by reference only. It does
not open a guard token; invoke native callables; capture, enqueue, allocate,
synchronize, or transfer; create or release an opaque handle; register
cleanup; or transition the READY lifecycle. Every rejection is read-only;
capture/replay own capture, handle lifetime, release, and cleanup.

**Parameters**

- `binding` (`object`) — Exact attached READY resident graph-capture binding.
- `prepared` (`object`) — Exact capture-ready enqueue prepared resident simulation.
- `capture_set` (`object`) — Exact resource preallocation published capture resource set.
- `adapter` (`GraphCaptureRuntimeAdapter`) — Caller-owned lazy native runtime adapter.

**Returns**

- (`PreparedGraphCaptureQualification`) — A fresh qualification retaining only exact existing metadata identities
- (`PreparedGraphCaptureQualification`) — and the adapter's exact callable vocabulary.

**Raises**

- `TypeError` — If a carrier, adapter member, probe result, or callable
record has an invalid exact type.
- `ValueError` — If READY binding metadata, identity compatibility, or
runtime/device/API qualification fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2070-L2277)

## particula.execution.graph_capture.renew_resident_graph_capture

*function*

```python
def renew_resident_graph_capture(binding: object, signature: object) -> GraphCaptureLifecycle
```

Explicitly renew a retired exact binding with a new request signature.

Renewal prepares ready metadata only; callers must separately declare
capture completion before the scheduler can admit the binding. It does not
recapture or replay a graph, replace resources, transfer data, synchronize,
or fall back.

**Parameters**

- `binding` (`object`) — Exact binding that owns retired lifecycle metadata.
- `signature` (`object`) — Exact new signature for the binding's retained request.

**Returns**

- (`GraphCaptureLifecycle`) — Ready lifecycle metadata now owned by ``binding``.

**Raises**

- `TypeError` — If ``binding`` or ``signature`` is inexact.
- `ValueError` — If retained identities differ or the lifecycle is not
retired.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2877-L2912)

## particula.execution.graph_capture.renew_retired_graph_capture

*function*

```python
def renew_retired_graph_capture(lifecycle: GraphCaptureLifecycle, signature: ResidentGraphCaptureSignature) -> GraphCaptureLifecycle
```

Prepare ready host metadata without recapture or session work.

This is the sole preparation path for a new declaration after retirement;
it does not create a native graph or replace the predecessor.

**Parameters**

- `lifecycle` (`GraphCaptureLifecycle`) — Retired lifecycle metadata to renew.
- `signature` (`ResidentGraphCaptureSignature`) — New identity signature to retain in the ready record.

**Returns**

- (`GraphCaptureLifecycle`) — A distinct ready lifecycle with the retired capability and new
- (`GraphCaptureLifecycle`) — signature identities.

**Raises**

- `TypeError` — If either argument has an inexact type.
- `ValueError` — If the lifecycle is not retired.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1382-L1415)

## particula.execution.graph_capture.replay_captured_resident_graph

*function*

```python
def replay_captured_resident_graph(captured: object, duration: object) -> None
```

Replay one authentic captured graph under exactly one resident token.

This direct-module-only entry point accepts only a capture-issued record
whose
opaque handle retains registered identity. After exact metadata admission,
it opens one token, invokes ``capture_launch`` once, and completes that
token once. Pinned array values and resident RNG words may change while
their retained identities remain compatible.

Accepted replay performs no prepared-host dispatch, native capture,
instantiation, release, allocation, runtime probe, payload scan, resource
Acquisition, transfer, readback, synchronization, RNG initialization or
reset, fallback, retry, or automatic recapture. Native-launch and
post-launch completion failures are writer-capable and fault resident and
capture metadata without a rollback guarantee.

**Parameters**

- `captured` (`object`) — Exact capture-issued ``CapturedResidentGraph`` to replay.
- `duration` (`object`) — Finite nonnegative replay duration matching the qualification.

**Raises**

- `TypeError` — If ``captured`` or retained metadata has an invalid exact
type, or ``duration`` is not a valid resident-step duration.
- `ValueError` — If the record is unauthentic, its handle or metadata
drifted, the binding cannot be admitted, or the duration mismatches.
- `BaseException` — Propagates native-launch or completion failure after
writer-capable cleanup and capture-failure classification.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2677-L2738)

## particula.execution.graph_capture.resolve_graph_capture_capability

*function*

```python
def resolve_graph_capture_capability(device: Device, probe: GraphCaptureRuntimeProbe) -> GraphCaptureCapability
```

Resolve graph-capture capability without importing a runtime.

CPU and Warp CPU declarations resolve without invoking ``probe``. Other
devices invoke the caller-owned runtime, device, and capture-API checks in
that order. Probe exceptions propagate unchanged.

**Parameters**

- `device` (`Device`) — Exact declared device to assess.
- `probe` (`GraphCaptureRuntimeProbe`) — Caller-provided lazy runtime probe.

**Returns**

- (`GraphCaptureCapability`) — Immutable capability declaration for ``device``.

**Raises**

- `TypeError` — If ``device`` is inexact, a probe method is not callable, or
a probe does not return a literal ``bool``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L224-L273)

## particula.execution.graph_capture.retire_graph_capture

*function*

```python
def retire_graph_capture(lifecycle: GraphCaptureLifecycle) -> GraphCaptureLifecycle
```

Retire invalidated host metadata without native graph or session work.

**Parameters**

- `lifecycle` (`GraphCaptureLifecycle`) — Invalidated lifecycle metadata to retire.

**Returns**

- (`GraphCaptureLifecycle`) — A retired successor, or the original retired record for repeated
- (`GraphCaptureLifecycle`) — retirement.

**Raises**

- `TypeError` — If ``lifecycle`` is not an exact lifecycle record.
- `ValueError` — If the lifecycle is not invalidated or retired.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L1354-L1379)

## particula.execution.graph_capture.retire_resident_graph_capture

*function*

```python
def retire_resident_graph_capture(binding: object) -> GraphCaptureLifecycle
```

Explicitly retire invalidated lifecycle metadata on one binding.

**Parameters**

- `binding` (`object`) — Exact binding that owns invalidated lifecycle metadata.

**Returns**

- (`GraphCaptureLifecycle`) — Retired lifecycle metadata now owned by ``binding``.

**Raises**

- `TypeError` — If ``binding`` is not an exact binding.
- `ValueError` — If the lifecycle is neither invalidated nor already
retired.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L2855-L2874)

## particula.execution.graph_capture.validate_resident_capture_resources

*function*

```python
def validate_resident_capture_resources(request: object) -> object
```

Resolve one published capture set with fixed identity checks only.

This lazy concrete helper is deliberately metadata-only. It performs the
registry's cached requirements lookup and confirms that the published set
and cached immutable report still describe the request-bound resident
resources. It does not prepare or acquire resources, rebuild accounting,
inspect payloads, or synchronize.

**Parameters**

- `request` (`object`) — Exact resident simulation request retaining published requirements.

**Returns**

- (`object`) — The exact registry-published capture resource set.

**Raises**

- `TypeError` — If ``request`` is not an exact resident simulation request.
- `ValueError` — If publication is absent, stale, or incompatible with the
request-bound capture resources.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/graph_capture.py#L468-L534)
