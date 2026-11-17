# particula.execution.cpu_simulation

*module*

Native NumPy fixed-step ownership, observation and exact CPU continuation.

This owner executes native CPU runnables with CPUExecutionContext, never Warp
CPU. Each complete cycle stages state, process diagnostics and RNG together.
Restart compares signatures of explicitly supported native configurations.
It does not infer custom code identity or impose GPU capability restrictions.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L1-L1)

## particula.execution.cpu_simulation.CPUCheckpoint

*class* · *dataclass*

```python
class CPUCheckpoint
```

Retain native carriers and canonical CPU random continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L66-L79)

### particula.execution.cpu_simulation.CPUCheckpoint.auxiliary

*attribute* · *instance attribute*

```python
auxiliary: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L79-L79)

### particula.execution.cpu_simulation.CPUCheckpoint.box_ids

*attribute* · *instance attribute*

```python
box_ids: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L77-L77)

### particula.execution.cpu_simulation.CPUCheckpoint.completed_steps

*attribute* · *instance attribute*

```python
completed_steps: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L75-L75)

### particula.execution.cpu_simulation.CPUCheckpoint.processes

*attribute* · *instance attribute*

```python
processes: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L72-L72)

### particula.execution.cpu_simulation.CPUCheckpoint.rng

*attribute* · *instance attribute*

```python
rng: bytes
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L78-L78)

### particula.execution.cpu_simulation.CPUCheckpoint.root_seed

*attribute* · *instance attribute*

```python
root_seed: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L76-L76)

### particula.execution.cpu_simulation.CPUCheckpoint.schema_version

*attribute* · *instance attribute*

```python
schema_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L70-L70)

### particula.execution.cpu_simulation.CPUCheckpoint.state

*attribute* · *instance attribute*

```python
state: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L71-L71)

### particula.execution.cpu_simulation.CPUCheckpoint.time

*attribute* · *instance attribute*

```python
time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L73-L73)

### particula.execution.cpu_simulation.CPUCheckpoint.time_step

*attribute* · *instance attribute*

```python
time_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L74-L74)

### particula.execution.cpu_simulation.CPUCheckpoint.__init__

*method*

```python
def __init__(schema_version: int, state: tuple, processes: tuple, time: float, time_step: float, completed_steps: int, root_seed: int, box_ids: tuple[str, ...], rng: bytes, auxiliary: tuple) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L0-L0)

## particula.execution.cpu_simulation.CPUSimulation

*class*

```python
class CPUSimulation
```

Own a detached CPU simulation with explicit duration and RNG identity.

Supported built-in configuration is compiled for identity only; numerical
execution always calls native CPU runnables. snapshot() is detached. Output
times may shorten cycles; exact continuation assumes the same subsequent
schedule and forcing. Native arrays/configuration supplied by the caller
may subsequently change without mutating this owner.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L82-L267)

### particula.execution.cpu_simulation.CPUSimulation._context

*attribute* · *instance attribute*

```python
_context = CPUExecutionContext(root_seed, ids)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L125-L125)

### particula.execution.cpu_simulation.CPUSimulation._plans

*attribute* · *instance attribute*

```python
_plans = tuple((cpu_process_identity(p) for p in leaves))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L109-L109)

### particula.execution.cpu_simulation.CPUSimulation._processes

*attribute* · *instance attribute*

```python
_processes = deepcopy(leaves)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L114-L114)

### particula.execution.cpu_simulation.CPUSimulation._state

*attribute* · *instance attribute*

```python
_state = _cpu_batch.detached(aerosol)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L115-L115)

### particula.execution.cpu_simulation.CPUSimulation._time

*attribute* · *instance attribute*

```python
_time = _cpu_batch.duration(time, 1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L116-L116)

### particula.execution.cpu_simulation.CPUSimulation.completed_steps

*attribute* · *instance attribute*

```python
completed_steps = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L127-L127)

### particula.execution.cpu_simulation.CPUSimulation.time_step

*attribute* · *instance attribute*

```python
time_step = _cpu_batch.duration(time_step, 1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L117-L117)

### particula.execution.cpu_simulation.CPUSimulation.time

*property*

```python
time
```

Return accepted absolute time [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L130-L132)

### particula.execution.cpu_simulation.CPUSimulation.__init__

*method*

```python
def __init__(aerosol, process, *, time_step, time=0.0, root_seed=0, box_ids=None)
```

Validate built-in processes and detach state/configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L92-L127)

### particula.execution.cpu_simulation.CPUSimulation._auxiliary

*method*

```python
def _auxiliary()
```

Detach last-step latent energy as a diagnostic.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L184-L198)

### particula.execution.cpu_simulation.CPUSimulation._restore_auxiliary

*method*

```python
def _restore_auxiliary(auxiliary)
```

Validate per-process diagnostic state before restore.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L200-L230)

### particula.execution.cpu_simulation.CPUSimulation.checkpoint

*method*

```python
def checkpoint()
```

Detach native state and RNG bytes between accepted cycles.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L169-L182)

### particula.execution.cpu_simulation.CPUSimulation.resume

*method* · *classmethod*

```python
def resume(checkpoint, process)
```

Restore identical declared process configurations and CPU RNG.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L232-L267)

### particula.execution.cpu_simulation.CPUSimulation.snapshot

*method*

```python
def snapshot()
```

Return detached native accepted state with auxiliary fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L165-L167)

### particula.execution.cpu_simulation.CPUSimulation.step

*method*

```python
def step(bound)
```

Accept a complete CPU cycle; retain state/RNG/clock on failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L134-L163)

## particula.execution.cpu_simulation.pack_native

*function*

```python
def pack_native(state)
```

Detach native carriers, auxiliary field schemas, values and budgets.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L21-L36)

## particula.execution.cpu_simulation.unpack_native

*function*

```python
def unpack_native(saved)
```

Validate and reconstruct native fields with explicit names and shapes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/cpu_simulation.py#L39-L63)
