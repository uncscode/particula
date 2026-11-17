# particula.execution.gpu_restart

*module*

Exact-device disk continuation for native adaptive GPU condensation.

Checkpointing explicitly downloads accepted scaled coordinates and tolerances.
Restart uploads those exact coordinates, not a multiply/divide reconstruction
from a physical snapshot. Trial scratch is recomputed on the next step.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L1-L1)

## particula.execution.gpu_restart.GPUCondensationCheckpoint

*class* · *dataclass*

```python
class GPUCondensationCheckpoint
```

Own accepted arrays, exact device identity and controller data.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L23-L45)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.atol

*attribute* · *instance attribute*

```python
atol: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L35-L35)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.base

*attribute* · *instance attribute*

```python
base: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L30-L30)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.configuration

*attribute* · *instance attribute*

```python
configuration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L31-L31)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.control_readbacks

*attribute* · *instance attribute*

```python
control_readbacks: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L44-L44)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.device

*attribute* · *instance attribute*

```python
device: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L29-L29)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.forcing

*attribute* · *instance attribute*

```python
forcing: EnvironmentForcing | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L45-L45)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.last_error

*attribute* · *instance attribute*

```python
last_error: float | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L42-L42)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.max_step

*attribute* · *instance attribute*

```python
max_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L40-L40)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.max_trials

*attribute* · *instance attribute*

```python
max_trials: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L41-L41)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.method

*attribute* · *instance attribute*

```python
method: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L37-L37)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.min_step

*attribute* · *instance attribute*

```python
min_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L39-L39)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.next_step

*attribute* · *instance attribute*

```python
next_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L38-L38)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.rtol

*attribute* · *instance attribute*

```python
rtol: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L34-L34)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.runtime

*attribute* · *instance attribute*

```python
runtime: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L28-L28)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.scale

*attribute* · *instance attribute*

```python
scale: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L33-L33)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.schema_version

*attribute* · *instance attribute*

```python
schema_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L27-L27)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.state

*attribute* · *instance attribute*

```python
state: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L32-L32)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.statistics

*attribute* · *instance attribute*

```python
statistics: StepStatistics
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L43-L43)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.time

*attribute* · *instance attribute*

```python
time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L36-L36)

### particula.execution.gpu_restart.GPUCondensationCheckpoint.__init__

*method*

```python
def __init__(schema_version: int, runtime: tuple, device: tuple, base: tuple, configuration: object, state: np.ndarray, scale: np.ndarray, rtol: np.ndarray, atol: np.ndarray, time: float, method: str, next_step: float, min_step: float, max_step: float, max_trials: int, last_error: float | None, statistics: StepStatistics, control_readbacks: int, forcing: EnvironmentForcing | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L0-L0)

## particula.execution.gpu_restart._device_identity

*function*

```python
def _device_identity(device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L52-L59)

## particula.execution.gpu_restart._runtime

*function*

```python
def _runtime()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L48-L49)

## particula.execution.gpu_restart._validate

*function*

```python
def _validate(saved)
```

Validate all host state before allocation or archive replacement.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L91-L117)

## particula.execution.gpu_restart._validate_arrays

*function*

```python
def _validate_arrays(saved, base)
```

Check scaled arrays and native inventory before any device upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L120-L165)

## particula.execution.gpu_restart._validate_controls

*function*

```python
def _validate_controls(saved)
```

Check the accepted clock, controller limits and work history.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L168-L193)

## particula.execution.gpu_restart.checkpoint_gpu_condensation

*function*

```python
def checkpoint_gpu_condensation(owner)
```

Explicitly download an accepted adaptive checkpoint outside capture.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L62-L88)

## particula.execution.gpu_restart.load_gpu_checkpoint

*function*

```python
def load_gpu_checkpoint(path, *, max_bytes=512 * 1024 * 1024)
```

Load validated adaptive host data without creating a device owner.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L252-L269)

## particula.execution.gpu_restart.resume_gpu_condensation

*function*

```python
def resume_gpu_condensation(saved, strategy, *, device)
```

Restore exact coordinates and controller on the same explicit device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L196-L230)

## particula.execution.gpu_restart.save_gpu_checkpoint

*function*

```python
def save_gpu_checkpoint(checkpoint, path, *, compressed=True)
```

Atomically store downloaded adaptive state as non-executable data.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_restart.py#L233-L249)
