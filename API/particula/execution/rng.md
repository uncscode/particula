# particula.execution.rng

*module*

Manage deterministic, caller-owned resident RNG stream state.

This concrete-only, direct-import module keeps host stream identity independent
of optional GPU dependencies. It derives reproducible initial words and
explicitly initializes caller-supplied Warp state arrays without acquiring,
replacing, or rebinding resources. Initialization allocates temporary host and
Warp copy sources only.

This module is intentionally not re-exported through ``particula.execution``
or the top-level package.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L1-L1)

## particula.execution.rng.MAX_LOGICAL_BOX_ID_BYTES

*attribute* · *module attribute*

```python
MAX_LOGICAL_BOX_ID_BYTES = 256
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L20-L20)

## particula.execution.rng.MAX_ROOT_SEED

*attribute* · *module attribute*

```python
MAX_ROOT_SEED = 2 ** 32 - 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L21-L21)

## particula.execution.rng.PROCESS_IDS

*attribute* · *module attribute*

```python
PROCESS_IDS = ('coagulation', 'wall_loss')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L22-L22)

## particula.execution.rng.STREAM_SCHEMA_VERSION

*attribute* · *module attribute*

```python
STREAM_SCHEMA_VERSION = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L19-L19)

## particula.execution.rng.SUPPORTED_PROCESS_IDS

*attribute* · *module attribute*

```python
SUPPORTED_PROCESS_IDS = PROCESS_IDS
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L23-L23)

## particula.execution.rng.StreamDescriptor

*class* · *dataclass*

```python
class StreamDescriptor
```

Associate a stream key with a physical caller-owned array lane.

**Parameters**

- `key` (`StreamKey`) — Valid immutable stream identity.
- `lane` (`int`) — Non-boolean integral physical lane. Registry admission checks its range against the registry dimensions.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L75-L97)

### particula.execution.rng.StreamDescriptor.key

*attribute* · *instance attribute*

```python
key: StreamKey
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L85-L85)

### particula.execution.rng.StreamDescriptor.lane

*attribute* · *instance attribute*

```python
lane: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L86-L86)

### particula.execution.rng.StreamDescriptor.__init__

*method*

```python
def __init__(key: StreamKey, lane: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L0-L0)

### particula.execution.rng.StreamDescriptor.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate descriptor carrier types without registry context.

**Raises**

- `TypeError` — If ``key`` is not a StreamKey or ``lane`` is not a
non-boolean integral value.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L88-L97)

## particula.execution.rng.StreamKey

*class* · *dataclass*

```python
class StreamKey
```

Identify one process-specific RNG stream for a logical box.

The logical-box identifier is retained exactly as supplied. It is not
normalized before deterministic initial-word derivation.

**Parameters**

- `schema_version` (`int`) — Supported stream-schema version.
- `process_id` (`str`) — Supported process namespace.
- `logical_box_id` (`str`) — Stable, unnormalized UTF-8 logical-box identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L38-L72)

### particula.execution.rng.StreamKey.logical_box_id

*attribute* · *instance attribute*

```python
logical_box_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L53-L53)

### particula.execution.rng.StreamKey.process_id

*attribute* · *instance attribute*

```python
process_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L52-L52)

### particula.execution.rng.StreamKey.schema_version

*attribute* · *instance attribute*

```python
schema_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L51-L51)

### particula.execution.rng.StreamKey.__init__

*method*

```python
def __init__(schema_version: int, process_id: str, logical_box_id: str) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L0-L0)

### particula.execution.rng.StreamKey.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate immutable stream identity metadata.

**Raises**

- `TypeError` — If a field has an unsupported type.
- `ValueError` — If the schema version, process, or logical ID is
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L55-L72)

## particula.execution.rng.StreamManifest

*class* · *dataclass*

```python
class StreamManifest
```

Expose immutable host-only stream identity metadata.

This carrier deliberately contains no state arrays, pointers, device values,
or current RNG words.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L100-L111)

### particula.execution.rng.StreamManifest.descriptors

*attribute* · *instance attribute*

```python
descriptors: tuple[StreamDescriptor, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L111-L111)

### particula.execution.rng.StreamManifest.lanes

*attribute* · *instance attribute*

```python
lanes: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L110-L110)

### particula.execution.rng.StreamManifest.logical_box_ids

*attribute* · *instance attribute*

```python
logical_box_ids: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L109-L109)

### particula.execution.rng.StreamManifest.root_seed

*attribute* · *instance attribute*

```python
root_seed: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L108-L108)

### particula.execution.rng.StreamManifest.__init__

*method*

```python
def __init__(root_seed: int, logical_box_ids: tuple[str, ...], lanes: tuple[int, ...], descriptors: tuple[StreamDescriptor, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L0-L0)

## particula.execution.rng.StreamRegistry

*class*

```python
class StreamRegistry
```

Retain one deterministic two-process state-array manifest by identity.

Construction performs host-only metadata and collision validation, without
importing optional dependencies or mutating state arrays. State-array
schema validation and copying are deferred to :meth:`initialize`. Derived
words are keyed only by the root seed and immutable stream identity, then
stored by each descriptor's physical lane. The canonical manifest has
independent coagulation and wall-loss namespaces for every logical box.

**Attributes**

- `root_seed` (`int`) — Validated root seed used for deterministic derivation.
- `n_boxes` (`int`) — Fixed number of physical state-array lanes.
- `logical_box_ids` (`tuple[str, ...]`) — Input-order immutable logical-box identifiers.
- `lanes` (`tuple[int, ...]`) — Input-order immutable physical lanes.
- `descriptors` (`tuple[StreamDescriptor, ...]`) — Canonical process-then-logical-ID stream descriptors.
- `state_arrays` (`tuple[tuple[str, Any], ...]`) — Canonical ordered caller-owned arrays, retained by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L196-L620)

### particula.execution.rng.StreamRegistry._descriptor_by_key

*attribute* · *instance attribute*

```python
_descriptor_by_key = {descriptor.key: descriptor for descriptor in self._descriptors}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L275-L277)

### particula.execution.rng.StreamRegistry._descriptors

*attribute* · *instance attribute*

```python
_descriptors = tuple((StreamDescriptor(StreamKey(STREAM_SCHEMA_VERSION, process_id, logical_box_id), lane) for process_id in PROCESS_IDS for (logical_box_id, lane) in zip(logical_box_ids, lanes, strict=True)))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L267-L274)

### particula.execution.rng.StreamRegistry._lane_by_id

*attribute* · *instance attribute*

```python
_lane_by_id = dict(zip(logical_box_ids, lanes, strict=True))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L278-L278)

### particula.execution.rng.StreamRegistry._lanes

*attribute* · *instance attribute*

```python
_lanes = lanes
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L265-L265)

### particula.execution.rng.StreamRegistry._logical_box_ids

*attribute* · *instance attribute*

```python
_logical_box_ids = logical_box_ids
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L264-L264)

### particula.execution.rng.StreamRegistry._n_boxes

*attribute* · *instance attribute*

```python
_n_boxes = int(n_boxes)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L263-L263)

### particula.execution.rng.StreamRegistry._root_seed

*attribute* · *instance attribute*

```python
_root_seed = int(root_seed)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L262-L262)

### particula.execution.rng.StreamRegistry._state_arrays

*attribute* · *instance attribute*

```python
_state_arrays = state_arrays
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L266-L266)

### particula.execution.rng.StreamRegistry._words_by_process

*attribute* · *instance attribute*

```python
_words_by_process = self._build_words_by_process()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L279-L279)

### particula.execution.rng.StreamRegistry.get_derived_word

*attribute* · *class attribute* · *instance attribute*

```python
get_derived_word = word_for
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L404-L404)

### particula.execution.rng.StreamRegistry.get_descriptor

*attribute* · *class attribute* · *instance attribute*

```python
get_descriptor = descriptor_for
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L360-L360)

### particula.execution.rng.StreamRegistry.get_lane

*attribute* · *class attribute* · *instance attribute*

```python
get_lane = lane_for
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L384-L384)

### particula.execution.rng.StreamRegistry.get_state_array

*attribute* · *class attribute* · *instance attribute*

```python
get_state_array = state_array_for
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L438-L438)

### particula.execution.rng.StreamRegistry.descriptors

*property*

```python
descriptors: tuple[StreamDescriptor, ...]
```

Return canonical process-then-logical-ID stream descriptors.

**Returns**

- (`tuple[StreamDescriptor, ...]`) — Immutable descriptors ordered by process, then input logical ID.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L318-L324)

### particula.execution.rng.StreamRegistry.lanes

*property*

```python
lanes: tuple[int, ...]
```

Return input-order immutable physical lanes.

**Returns**

- (`tuple[int, ...]`) — Exact input-order tuple of physical state-array lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L309-L315)

### particula.execution.rng.StreamRegistry.logical_box_ids

*property*

```python
logical_box_ids: tuple[str, ...]
```

Return input-order immutable logical IDs.

**Returns**

- (`tuple[str, ...]`) — Exact logical-ID tuple retained during construction.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L300-L306)

### particula.execution.rng.StreamRegistry.n_boxes

*property*

```python
n_boxes: int
```

Return the fixed physical state-array size.

**Returns**

- (`int`) — Number of physical lanes in each retained state array.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L291-L297)

### particula.execution.rng.StreamRegistry.root_seed

*property*

```python
root_seed: int
```

Return the validated root seed.

**Returns**

- (`int`) — Root seed used for all registry stream derivations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L282-L288)

### particula.execution.rng.StreamRegistry.state_arrays

*property*

```python
state_arrays: tuple[tuple[str, Any], ...]
```

Return retained ordered caller-owned arrays by identity.

**Returns**

- (`tuple[tuple[str, Any], ...]`) — Canonical process-and-array manifest supplied at construction.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L327-L333)

### particula.execution.rng.StreamRegistry.__init__

*method*

```python
def __init__(root_seed: int, n_boxes: int, logical_box_ids: tuple[str, ...], lanes: tuple[int, ...], state_arrays: tuple[tuple[str, Any], ...]) -> None
```

Validate and retain an immutable host-only stream manifest.

**Parameters**

- `root_seed` (`int`) — Non-boolean unsigned 32-bit root seed.
- `n_boxes` (`int`) — Non-boolean number of fixed physical lanes.
- `logical_box_ids` (`tuple[str, ...]`) — Exact input-order tuple of unique logical IDs.
- `lanes` (`tuple[int, ...]`) — Exact tuple mapping input-order IDs to physical lanes.
- `state_arrays` (`tuple[tuple[str, Any], ...]`) — Exact canonical two-process tuple of caller arrays.

**Raises**

- `TypeError` — If manifest metadata does not have the required types.
- `ValueError` — If metadata is out of range, malformed, noncanonical,
or produces a same-process derived-word collision.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L216-L279)

### particula.execution.rng.StreamRegistry._build_words_by_process

*method*

```python
def _build_words_by_process() -> dict[str, tuple[int, ...]]
```

Build lane-indexed words and reject same-process collisions.

**Returns**

- (`dict[str, tuple[int, ...]]`) — Process-indexed immutable tuples of words by physical lane.

**Raises**

- `ValueError` — If two input-order logical IDs derive the same word for
one process.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L515-L545)

### particula.execution.rng.StreamRegistry._validate_state_arrays

*method*

```python
def _validate_state_arrays() -> None
```

Validate retained Warp arrays fully before initialization writes.

**Raises**

- `ImportError` — If the optional Warp dependency is unavailable.
- `TypeError` — If a retained array is not a valid Warp uint32 array.
- `ValueError` — If an array has an invalid layout, device, identity, or
memory-alias relationship.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L547-L566)

### particula.execution.rng.StreamRegistry.descriptor_for

*method*

```python
def descriptor_for(process_id: str, logical_box_id: str) -> StreamDescriptor
```

Return the descriptor for an exact registered process and ID.

**Parameters**

- `process_id` (`str`) — Supported process namespace.
- `logical_box_id` (`str`) — Exact registered logical-box identifier.

**Returns**

- (`StreamDescriptor`) — Immutable descriptor associated with the requested stream.

**Raises**

- `TypeError` — If either identity component has an invalid type.
- `ValueError` — If the process or logical ID is invalid.
- `LookupError` — If the valid stream identity is not registered.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L335-L358)

### particula.execution.rng.StreamRegistry.initialize

*method*

```python
def initialize() -> None
```

Overwrite retained arrays with deterministic initial words.

NumPy and Warp are imported only after host manifest preflight. The
temporary host and Warp copy sources do not change state-array
ownership: this method deterministically overwrites retained caller
arrays without acquiring, replacing, or rebinding them. Preflight
failures write neither buffer. It copies lane-indexed words for
coagulation first and wall loss second. A device or copy failure after
the first successful copy has no rollback guarantee; callers may correct
the failure and retry without replacing this registry's arrays.

**Raises**

- `ImportError` — If NumPy or Warp is unavailable.
- `TypeError` — If a retained state array has an invalid type or dtype.
- `ValueError` — If retained arrays have invalid schemas, devices, or
aliasing relationships.
- `RuntimeError` — If Warp reports a device-copy failure after
validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L568-L591)

### particula.execution.rng.StreamRegistry.initialize_process

*method*

```python
def initialize_process(process_id: str) -> None
```

Initialize exactly one validated process state array.

This narrow internal primitive lets a resident resource publish a new
process sidecar without reseeding an already-published sibling stream,
such as wall loss after coagulation.

**Parameters**

- `process_id` (`str`) — Supported process namespace to initialize.

**Raises**

- `TypeError` — If the process ID or either manifest array is invalid.
- `ValueError` — If either retained array has invalid schema, device, or
aliasing metadata.
- `RuntimeError` — If Warp reports a device-copy failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L593-L620)

### particula.execution.rng.StreamRegistry.initialize_selected

*method*

```python
def initialize_selected(process_ids: tuple[str, ...] | None = None, logical_box_ids: tuple[str, ...] | None = None) -> None
```

Reinitialize only selected process and logical-box stream lanes.

Full retained-array schema and nonaliasing preflight always precedes a
writer, including for an explicitly empty selection.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L449-L499)

### particula.execution.rng.StreamRegistry.inspect

*method*

```python
def inspect() -> StreamManifest
```

Return this registry's immutable host-only stream manifest.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L440-L447)

### particula.execution.rng.StreamRegistry.lane_for

*method*

```python
def lane_for(logical_box_id: str) -> int
```

Return the physical lane for an exact registered logical ID.

**Parameters**

- `logical_box_id` (`str`) — Exact registered logical-box identifier.

**Returns**

- (`int`) — Physical state-array lane for the logical ID.

**Raises**

- `TypeError` — If ``logical_box_id`` is not a string.
- `ValueError` — If the logical ID is malformed.
- `LookupError` — If the valid logical ID is not registered.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L362-L382)

### particula.execution.rng.StreamRegistry.preflight_selected

*method*

```python
def preflight_selected(process_ids: tuple[str, ...] | None = None, logical_box_ids: tuple[str, ...] | None = None) -> None
```

Validate a selected reset without allocating or launching writers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L501-L513)

### particula.execution.rng.StreamRegistry.state_array_for

*method*

```python
def state_array_for(process_id: str) -> Any
```

Return the retained caller-owned process state array by identity.

**Parameters**

- `process_id` (`str`) — Supported process namespace.

**Returns**

- (`Any`) — Original caller-owned state array for the requested process.

**Raises**

- `TypeError` — If ``process_id`` is not a string.
- `ValueError` — If ``process_id`` is unsupported.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L422-L436)

### particula.execution.rng.StreamRegistry.word_for

*method*

```python
def word_for(process_id: str, logical_box_id: str) -> int
```

Return the derived word for an exact registered stream.

**Parameters**

- `process_id` (`str`) — Supported process namespace.
- `logical_box_id` (`str`) — Exact registered logical-box identifier.

**Returns**

- (`int`) — Deterministic unsigned 32-bit initial word for the stream.

**Raises**

- `TypeError` — If either identity component has an invalid type.
- `ValueError` — If the process or logical ID is invalid.
- `LookupError` — If the valid stream identity is not registered.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L386-L402)

### particula.execution.rng.StreamRegistry.words_by_lane

*method*

```python
def words_by_lane(process_id: str) -> tuple[int, ...]
```

Return immutable derived words indexed by physical lane.

**Parameters**

- `process_id` (`str`) — Supported process namespace.

**Returns**

- (`tuple[int, ...]`) — Lane-indexed deterministic initial words for the process.

**Raises**

- `TypeError` — If ``process_id`` is not a string.
- `ValueError` — If ``process_id`` is unsupported.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/rng.py#L406-L420)
