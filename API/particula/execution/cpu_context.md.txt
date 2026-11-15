# particula.execution.cpu_context

*module*

Caller-owned, transactional random streams for CPU box execution.

This module does not import Warp or share the resident GPU stream format.
Context views share continuation state, while transactions stage independent
generators until a successful runnable substep publishes them.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L1-L1)

## particula.execution.cpu_context.CPUExecutionContext

*class*

```python
class CPUExecutionContext
```

Retain persistent process-specific streams keyed by logical box ID.

**Parameters**

- `root_seed` (`int`) — Nonnegative integer seed; booleans are not accepted.
- `box_ids` (`tuple[str, ...]`) — Unique nonempty strings in current numeric-array box order.

Reordering numeric storage requires a matching ``with_box_ids`` view.
Reusing a context and process namespace deliberately continues its streams.
Create a new context to reset them. Contexts are not thread-safe and must
not be used concurrently. No CPU/GPU bitwise equivalence is implied.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L36-L153)

### particula.execution.cpu_context.CPUExecutionContext._box_ids

*attribute* · *instance attribute*

```python
_box_ids: tuple[str, ...] = _box_ids(box_ids)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L58-L58)

### particula.execution.cpu_context.CPUExecutionContext._root_seed

*attribute* · *instance attribute*

```python
_root_seed: int = int(root_seed)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L57-L57)

### particula.execution.cpu_context.CPUExecutionContext._streams

*attribute* · *instance attribute*

```python
_streams: dict[tuple[str, str], np.random.Generator] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L59-L59)

### particula.execution.cpu_context.CPUExecutionContext.box_ids

*property*

```python
box_ids: tuple[str, ...]
```

Return logical IDs in this view's numeric-array order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L67-L69)

### particula.execution.cpu_context.CPUExecutionContext.root_seed

*property*

```python
root_seed: int
```

Return the immutable root seed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L62-L64)

### particula.execution.cpu_context.CPUExecutionContext.__init__

*method*

```python
def __init__(root_seed: int, box_ids: tuple[str, ...]) -> None
```

Validate metadata without creating or advancing random streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L49-L59)

### particula.execution.cpu_context.CPUExecutionContext._begin

*method*

```python
def _begin(n_boxes: int) -> _CPUStreamTransaction
```

Validate layout and begin a private substep transaction.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L81-L85)

### particula.execution.cpu_context.CPUExecutionContext._checkpoint

*method*

```python
def _checkpoint() -> bytes
```

Own canonical JSON bytes for exact PCG64 continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L87-L100)

### particula.execution.cpu_context.CPUExecutionContext._new_generator

*method*

```python
def _new_generator(namespace: str, box_id: str) -> np.random.Generator
```

Derive a versioned PCG64 stream independently of lane/order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L143-L153)

### particula.execution.cpu_context.CPUExecutionContext._restore

*method* · *classmethod*

```python
def _restore(payload: bytes, *, root_seed, box_ids, namespaces)
```

Validate continuation records before creating generators.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L102-L141)

### particula.execution.cpu_context.CPUExecutionContext.with_box_ids

*method*

```python
def with_box_ids(box_ids: tuple[str, ...]) -> CPUExecutionContext
```

Create a reordered, reduced, or extended continuation-sharing view.

New IDs acquire streams lazily; existing IDs retain continuation even
when absent from a view. The caller must reorder arrays consistently.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L71-L79)

## particula.execution.cpu_context._CPUStreamTransaction

*class*

```python
class _CPUStreamTransaction
```

Stage stream continuation without touching caller-owned state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L156-L179)

### particula.execution.cpu_context._CPUStreamTransaction._context

*attribute* · *instance attribute*

```python
_context = context
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L160-L160)

### particula.execution.cpu_context._CPUStreamTransaction._staged

*attribute* · *instance attribute*

```python
_staged: dict[tuple[str, str], np.random.Generator] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L161-L161)

### particula.execution.cpu_context._CPUStreamTransaction.__init__

*method*

```python
def __init__(context: CPUExecutionContext) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L159-L161)

### particula.execution.cpu_context._CPUStreamTransaction.commit

*method*

```python
def commit() -> None
```

Publish staged streams after successful numeric publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L177-L179)

### particula.execution.cpu_context._CPUStreamTransaction.generator

*method*

```python
def generator(namespace: str, box: int) -> np.random.Generator
```

Return a staged generator for one process and logical box.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L163-L175)

## particula.execution.cpu_context._box_ids

*function*

```python
def _box_ids(values: tuple[str, ...]) -> tuple[str, ...]
```

Validate stable, unique logical identities without normalization.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L25-L33)

## particula.execution.cpu_context._extension_rng_namespace

*function*

```python
def _extension_rng_namespace(extension_id: str, name: str) -> str
```

Encode identity components without delimiter collisions.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L18-L22)

## particula.execution.cpu_context._validate_pcg64_state

*function*

```python
def _validate_pcg64_state(state)
```

Reject coercible or out-of-range words before NumPy state assignment.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L182-L204)
