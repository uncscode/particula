# particula.execution.gpu_coupled_restart

*module*

Exact boundary restart for fixed GPU events and adaptive condensation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L1-L1)

## particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint

*class* · *dataclass*

```python
class GPUCoupledCheckpoint
```

Save event/RNG bytes and the complete deterministic boundary policy.

No in-progress adaptive controller is carried across a fixed event boundary:
every cycle deliberately resets it from the new accepted physical state.
Future forcing and source schedules are supplied again by the caller.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L25-L43)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.configuration

*attribute* · *instance attribute*

```python
configuration: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L38-L38)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.controls

*attribute* · *instance attribute*

```python
controls: GPUCondensationControls
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L39-L39)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.counters

*attribute* · *instance attribute*

```python
counters: tuple[int, int, int, int]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L42-L42)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.device

*attribute* · *instance attribute*

```python
device: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L36-L36)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.events

*attribute* · *instance attribute*

```python
events: SimulationCheckpoint
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L37-L37)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.order

*attribute* · *instance attribute*

```python
order: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L40-L40)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.resident

*attribute* · *instance attribute*

```python
resident: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L43-L43)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.runtime

*attribute* · *instance attribute*

```python
runtime: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L35-L35)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.schema_version

*attribute* · *instance attribute*

```python
schema_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L34-L34)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.statistics

*attribute* · *instance attribute*

```python
statistics: StepStatistics
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L41-L41)

### particula.execution.gpu_coupled_restart.GPUCoupledCheckpoint.__init__

*method*

```python
def __init__(schema_version: int, runtime: tuple, device: tuple, events: SimulationCheckpoint, configuration: object, controls: GPUCondensationControls, order: str, statistics: StepStatistics, counters: tuple[int, int, int, int], resident: bool) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L0-L0)

## particula.execution.gpu_coupled_restart._validate

*function*

```python
def _validate(saved)
```

Check the complete host envelope without allocating device state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L71-L127)

## particula.execution.gpu_coupled_restart.checkpoint_coupled

*function*

```python
def checkpoint_coupled(owner)
```

Download exact continuation only after a successfully completed cycle.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L46-L68)

## particula.execution.gpu_coupled_restart.load_coupled_checkpoint

*function*

```python
def load_coupled_checkpoint(path, *, max_bytes=512 * 1024 * 1024)
```

Load validated host records without physics or CUDA allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L208-L225)

## particula.execution.gpu_coupled_restart.resume_coupled

*function*

```python
def resume_coupled(saved, events, strategy, *, device)
```

Restore exact event streams and controls on the saved explicit device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L130-L167)

## particula.execution.gpu_coupled_restart.save_coupled_checkpoint

*function*

```python
def save_coupled_checkpoint(saved, path, *, compressed=True)
```

Atomically save canonical host bytes without pickle or uploads.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/gpu_coupled_restart.py#L170-L205)
