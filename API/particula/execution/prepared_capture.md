# particula.execution.prepared_capture

*module*

Private, owner-bound CUDA capture for ordered subsets of prepared physics.

The complete canonical resident scheduler keeps its stricter public graph
qualification contract. This owner exposes no transferable graph handle: only
its own capture method can install a graph, and step/run remain the sole replay
entry points. Both paths use the resident guard and native-operation leases.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L1-L1)

## particula.execution.prepared_capture._OwnedGraph

*class* · *dataclass*

```python
class _OwnedGraph
```

Keep native storage private and pinned until its stream has drained.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L74-L107)

### particula.execution.prepared_capture._OwnedGraph.arrays

*attribute* · *instance attribute*

```python
arrays: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L81-L81)

### particula.execution.prepared_capture._OwnedGraph.graph

*attribute* · *instance attribute*

```python
graph: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L79-L79)

### particula.execution.prepared_capture._OwnedGraph.identities

*attribute* · *instance attribute*

```python
identities: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L80-L80)

### particula.execution.prepared_capture._OwnedGraph.owner

*attribute* · *instance attribute*

```python
owner: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L78-L78)

### particula.execution.prepared_capture._OwnedGraph.valid

*attribute* · *class attribute* · *instance attribute*

```python
valid: bool = True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L82-L82)

### particula.execution.prepared_capture._OwnedGraph.__init__

*method*

```python
def __init__(owner: Any, graph: Any, identities: tuple[Any, ...], arrays: tuple[Any, ...], valid: bool = True) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L0-L0)

### particula.execution.prepared_capture._OwnedGraph.invalidate

*method*

```python
def invalidate()
```

Revoke replay without releasing storage used by queued work.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L84-L86)

### particula.execution.prepared_capture._OwnedGraph.validate

*method*

```python
def validate(owner)
```

Reject stale bindings before graph launch or random draws.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L88-L107)

## particula.execution.prepared_capture._admit_capture

*function*

```python
def _admit_capture(owner)
```

Check supported target, binding and stream before capture work.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L110-L121)

## particula.execution.prepared_capture._signature

*function*

```python
def _signature(owner)
```

Retain identities and array descriptors without reading device values.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L13-L71)

## particula.execution.prepared_capture.capture

*function*

```python
def capture(owner)
```

Capture launches only; recording performs no step or scientific write.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L124-L162)

## particula.execution.prepared_capture.replay

*function*

```python
def replay(owner)
```

Launch only this owner's issued graph inside its single step token.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_capture.py#L165-L175)
