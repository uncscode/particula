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

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L29-L90)

### particula.execution.cpu_context.CPUExecutionContext._box_ids

*attribute* · *instance attribute*

```python
_box_ids: tuple[str, ...] = _box_ids(box_ids)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L51-L51)

### particula.execution.cpu_context.CPUExecutionContext._root_seed

*attribute* · *instance attribute*

```python
_root_seed: int = int(root_seed)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L50-L50)

### particula.execution.cpu_context.CPUExecutionContext._streams

*attribute* · *instance attribute*

```python
_streams: dict[tuple[str, str], np.random.Generator] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L52-L52)

### particula.execution.cpu_context.CPUExecutionContext.box_ids

*property*

```python
box_ids: tuple[str, ...]
```

Return logical IDs in this view's numeric-array order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L60-L62)

### particula.execution.cpu_context.CPUExecutionContext.root_seed

*property*

```python
root_seed: int
```

Return the immutable root seed.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L55-L57)

### particula.execution.cpu_context.CPUExecutionContext.__init__

*method*

```python
def __init__(root_seed: int, box_ids: tuple[str, ...]) -> None
```

Validate metadata without creating or advancing random streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L42-L52)

### particula.execution.cpu_context.CPUExecutionContext._begin

*method*

```python
def _begin(n_boxes: int) -> _CPUStreamTransaction
```

Validate layout and begin a private substep transaction.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L74-L78)

### particula.execution.cpu_context.CPUExecutionContext._new_generator

*method*

```python
def _new_generator(namespace: str, box_id: str) -> np.random.Generator
```

Derive a versioned PCG64 stream independently of lane/order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L80-L90)

### particula.execution.cpu_context.CPUExecutionContext.with_box_ids

*method*

```python
def with_box_ids(box_ids: tuple[str, ...]) -> CPUExecutionContext
```

Create a reordered, reduced, or extended continuation-sharing view.

New IDs acquire streams lazily; existing IDs retain continuation even
when absent from a view. The caller must reorder arrays consistently.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L64-L72)

## particula.execution.cpu_context._CPUStreamTransaction

*class*

```python
class _CPUStreamTransaction
```

Stage stream continuation without touching caller-owned state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L93-L116)

### particula.execution.cpu_context._CPUStreamTransaction._context

*attribute* · *instance attribute*

```python
_context = context
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L97-L97)

### particula.execution.cpu_context._CPUStreamTransaction._staged

*attribute* · *instance attribute*

```python
_staged: dict[tuple[str, str], np.random.Generator] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L98-L98)

### particula.execution.cpu_context._CPUStreamTransaction.__init__

*method*

```python
def __init__(context: CPUExecutionContext) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L96-L98)

### particula.execution.cpu_context._CPUStreamTransaction.commit

*method*

```python
def commit() -> None
```

Publish staged streams after successful numeric publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L114-L116)

### particula.execution.cpu_context._CPUStreamTransaction.generator

*method*

```python
def generator(namespace: str, box: int) -> np.random.Generator
```

Return a staged generator for one process and logical box.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L100-L112)

## particula.execution.cpu_context._box_ids

*function*

```python
def _box_ids(values: tuple[str, ...]) -> tuple[str, ...]
```

Validate stable, unique logical identities without normalization.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_context.py#L18-L26)
