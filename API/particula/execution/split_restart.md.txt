# particula.execution.split_restart

*module*

Complete-cycle CPU split restart with caller-supplied compatible flows.

Only the known condensation/dilution flow factories have a disk identity.
Neither callbacks nor live strategies are serialized or imported from a file.
Component controllers restart at every split boundary, so a complete-cycle
checkpoint needs no hidden half-step or component-controller state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L1-L1)

## particula.execution.split_restart.SplitCheckpoint

*class* · *dataclass*

```python
class SplitCheckpoint
```

Own native accepted state, nested flow identity and the outer clock.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L22-L31)

### particula.execution.split_restart.SplitCheckpoint.completed_steps

*attribute* · *instance attribute*

```python
completed_steps: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L31-L31)

### particula.execution.split_restart.SplitCheckpoint.configuration

*attribute* · *instance attribute*

```python
configuration: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L28-L28)

### particula.execution.split_restart.SplitCheckpoint.schema_version

*attribute* · *instance attribute*

```python
schema_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L26-L26)

### particula.execution.split_restart.SplitCheckpoint.state

*attribute* · *instance attribute*

```python
state: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L27-L27)

### particula.execution.split_restart.SplitCheckpoint.time

*attribute* · *instance attribute*

```python
time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L29-L29)

### particula.execution.split_restart.SplitCheckpoint.time_step

*attribute* · *instance attribute*

```python
time_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L30-L30)

### particula.execution.split_restart.SplitCheckpoint.__init__

*method*

```python
def __init__(schema_version: int, state: tuple, configuration: tuple, time: float, time_step: float, completed_steps: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L0-L0)

## particula.execution.split_restart._identity

*function*

```python
def _identity(node)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L34-L60)

## particula.execution.split_restart.checkpoint_split

*function*

```python
def checkpoint_split(owner)
```

Capture a detached complete accepted cycle and its declared ordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L63-L74)

## particula.execution.split_restart.resume_split

*function*

```python
def resume_split(saved, group)
```

Resume with an explicitly supplied matching nested group and controls.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/split_restart.py#L77-L96)
