# particula.execution.cpu_restart

*module*

Native CPU/adaptive disk restart with data records and caller physics.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L1-L1)

## particula.execution.cpu_restart.CondensationCheckpoint

*class* · *dataclass*

```python
class CondensationCheckpoint
```

Keep scaled state, controller history and native configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L22-L42)

### particula.execution.cpu_restart.CondensationCheckpoint.atol

*attribute* · *instance attribute*

```python
atol: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L34-L34)

### particula.execution.cpu_restart.CondensationCheckpoint.base

*attribute* · *instance attribute*

```python
base: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L27-L27)

### particula.execution.cpu_restart.CondensationCheckpoint.configuration

*attribute* · *instance attribute*

```python
configuration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L28-L28)

### particula.execution.cpu_restart.CondensationCheckpoint.forcing

*attribute* · *instance attribute*

```python
forcing: EnvironmentForcing | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L42-L42)

### particula.execution.cpu_restart.CondensationCheckpoint.last_error

*attribute* · *instance attribute*

```python
last_error: float | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L39-L39)

### particula.execution.cpu_restart.CondensationCheckpoint.max_step

*attribute* · *instance attribute*

```python
max_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L37-L37)

### particula.execution.cpu_restart.CondensationCheckpoint.max_trials

*attribute* · *instance attribute*

```python
max_trials: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L38-L38)

### particula.execution.cpu_restart.CondensationCheckpoint.method

*attribute* · *instance attribute*

```python
method: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L32-L32)

### particula.execution.cpu_restart.CondensationCheckpoint.min_step

*attribute* · *instance attribute*

```python
min_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L36-L36)

### particula.execution.cpu_restart.CondensationCheckpoint.next_step

*attribute* · *instance attribute*

```python
next_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L35-L35)

### particula.execution.cpu_restart.CondensationCheckpoint.rtol

*attribute* · *instance attribute*

```python
rtol: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L33-L33)

### particula.execution.cpu_restart.CondensationCheckpoint.scale

*attribute* · *instance attribute*

```python
scale: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L30-L30)

### particula.execution.cpu_restart.CondensationCheckpoint.schema_version

*attribute* · *instance attribute*

```python
schema_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L26-L26)

### particula.execution.cpu_restart.CondensationCheckpoint.state

*attribute* · *instance attribute*

```python
state: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L29-L29)

### particula.execution.cpu_restart.CondensationCheckpoint.statistics

*attribute* · *instance attribute*

```python
statistics: StepStatistics
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L40-L40)

### particula.execution.cpu_restart.CondensationCheckpoint.time

*attribute* · *instance attribute*

```python
time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L31-L31)

### particula.execution.cpu_restart.CondensationCheckpoint.viscosity

*attribute* · *instance attribute*

```python
viscosity: np.ndarray | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L41-L41)

### particula.execution.cpu_restart.CondensationCheckpoint.__init__

*method*

```python
def __init__(schema_version: int, base: tuple, configuration: object, state: np.ndarray, scale: np.ndarray, time: float, method: str, rtol: np.ndarray, atol: np.ndarray, next_step: float, min_step: float, max_step: float, max_trials: int, last_error: float | None, statistics: StepStatistics, viscosity: np.ndarray | None, forcing: EnvironmentForcing | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L0-L0)

## particula.execution.cpu_restart.checkpoint_condensation

*function*

```python
def checkpoint_condensation(owner)
```

Capture adaptive state without advancing or resetting its controller.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L45-L70)

## particula.execution.cpu_restart.load_cpu_checkpoint

*function*

```python
def load_cpu_checkpoint(path, *, max_bytes=512 * 1024 * 1024)
```

Read CPU restart data; resume separately with matching physics.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L147-L173)

## particula.execution.cpu_restart.resume_condensation

*function*

```python
def resume_condensation(saved, strategy)
```

Restore the same scaled coordinates and adaptive continuation policy.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L73-L117)

## particula.execution.cpu_restart.save_cpu_checkpoint

*function*

```python
def save_cpu_checkpoint(checkpoint, path, *, compressed=True)
```

Atomically save supported native CPU or adaptive controller data only.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_restart.py#L120-L144)
