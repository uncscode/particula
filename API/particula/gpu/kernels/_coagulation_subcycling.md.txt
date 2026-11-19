# particula.gpu.kernels._coagulation_subcycling

*module*

Bounded, live-state work admission for native coagulation intervals.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L1-L1)

## particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval

*class* · *dataclass*

```python
class AdaptiveInterval
```

Own the duration, admission history and ordered collision log.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L157-L325)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.config

*attribute* · *instance attribute*

```python
config: CoagulationSubcyclingConfig = config
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L172-L172)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.dt

*attribute* · *instance attribute*

```python
dt: Any = wp.zeros_like(self.remaining)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L182-L182)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.duration

*attribute* · *instance attribute*

```python
duration: float = float(duration)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L173-L173)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.elapsed

*attribute* · *instance attribute*

```python
elapsed: Any = wp.zeros(2, dtype=wp.float64, device=dev)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L181-L181)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.history

*attribute* · *instance attribute*

```python
history: Any = wp.zeros((k, b, 10), dtype=wp.float64, device=dev)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L184-L184)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.iteration

*attribute* · *instance attribute*

```python
iteration: Any = wp.zeros(1, dtype=wp.int32, device=dev)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L183-L183)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.offsets

*attribute* · *instance attribute*

```python
offsets: Any = wp.zeros(b, dtype=wp.int32, device=dev)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L186-L186)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.pairs

*attribute* · *instance attribute*

```python
pairs: Any = wp.zeros((b, max(1, n - 1), 2), dtype=wp.int32, device=dev)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L185-L185)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.remaining

*attribute* · *instance attribute*

```python
remaining: Any = wp.zeros(1, dtype=wp.float64, device=dev)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L180-L180)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.__init__

*method*

```python
def __init__(call, config, duration)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L171-L186)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.begin

*method*

```python
def begin()
```

Reset progress on the current stream before the first pass.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L188-L193)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.choose

*method*

```python
def choose(call, failure)
```

Admit one shared duration against all live box majorants.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L195-L219)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.diagnostics

*method*

```python
def diagnostics(call)
```

Download per-pass rows delimiting the ordered pair log.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L252-L279)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.enqueue

*method*

```python
def enqueue(call, failure, operation)
```

Record the bounded pass loop, including per-merge dependents.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L244-L250)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.finish

*method*

```python
def finish(call, failure)
```

Append the accepted work and consume its physical duration.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L221-L242)

### particula.gpu.kernels._coagulation_subcycling.AdaptiveInterval.rejection

*method*

```python
def rejection(call, code)
```

Keep tentative diagnostics independent of the restore point.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L281-L325)

## particula.gpu.kernels._coagulation_subcycling.CoagulationSubcyclingConfig

*class* · *dataclass*

```python
class CoagulationSubcyclingConfig
```

Set the minimum duration [s], maximum passes and memory ceiling [B].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L15-L35)

### particula.gpu.kernels._coagulation_subcycling.CoagulationSubcyclingConfig.max_device_bytes

*attribute* · *instance attribute*

```python
max_device_bytes: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L21-L21)

### particula.gpu.kernels._coagulation_subcycling.CoagulationSubcyclingConfig.max_subcycles

*attribute* · *instance attribute*

```python
max_subcycles: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L20-L20)

### particula.gpu.kernels._coagulation_subcycling.CoagulationSubcyclingConfig.min_substep

*attribute* · *instance attribute*

```python
min_substep: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L19-L19)

### particula.gpu.kernels._coagulation_subcycling.CoagulationSubcyclingConfig.__init__

*method*

```python
def __init__(min_substep: float, max_subcycles: int, max_device_bytes: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L0-L0)

### particula.gpu.kernels._coagulation_subcycling.CoagulationSubcyclingConfig.__post_init__

*method*

```python
def __post_init__()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L23-L35)

## particula.gpu.kernels._coagulation_subcycling.CoagulationSubcyclingError

*class*

```python
class CoagulationSubcyclingError(CoagulationBudgetError)
```

Bases: `particula.gpu.kernels._coagulation_budget.CoagulationBudgetError`

A recoverable outer interval rejection carrying tentative diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L328-L331)

### particula.gpu.kernels._coagulation_subcycling.CoagulationSubcyclingError.diagnostics

*attribute* · *instance attribute*

```python
diagnostics: dict[str, Any]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L331-L331)

## particula.gpu.kernels._coagulation_subcycling._choose

*function*

```python
def _choose(counts: wp.array(dtype=wp.int32), indices: Any, weights: Any, volumes: Any, majorants: Any, trial_limit: int, capacity: int, minimum: wp.float64, remaining: Any, dt: Any, iteration: Any, history: Any, report: Any, failure: Any)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L38-L104)

## particula.gpu.kernels._coagulation_subcycling._end

*function*

```python
def _end(failure: Any)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L150-L154)

## particula.gpu.kernels._coagulation_subcycling._finish

*function*

```python
def _finish(remaining: Any, dt: Any, iteration: Any, history: Any, pairs: Any, accepted: Any, work: Any, log: Any, offsets: Any, failure: Any, tolerance: wp.float64, duration: wp.float64, elapsed: Any)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L107-L147)

## particula.gpu.kernels._coagulation_subcycling.adaptive_capacity

*function*

```python
def adaptive_capacity(boxes, slots, byte_limit)
```

Choose the largest physical single-pass capacity permitted by bytes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L334-L339)

## particula.gpu.kernels._coagulation_subcycling.check_memory

*function*

```python
def check_memory(config, categories)
```

Reject the complete reservation before creating adaptive allocations.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L385-L392)

## particula.gpu.kernels._coagulation_subcycling.interval_bytes

*function*

```python
def interval_bytes(boxes, slots, passes)
```

Return bytes for controller scalars, per-box logs and diagnostic rows.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L342-L344)

## particula.gpu.kernels._coagulation_subcycling.workspace_bytes

*function*

```python
def workspace_bytes(boxes, slots, capacity, epsilon=None, polarizable=False)
```

Upper bound for the direct sampler with all optional transport arrays.

Six fp64 slot properties, two int32 slot indices, fp64 reduction lanes,
five environment inputs, mean density, rate, trial report, output count,
active count, RNG word/init flag and two work counters per box. Include
validation flags and both the persistent and setup failure latch.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L363-L382)

## particula.gpu.kernels._coagulation_subcycling.wrap_subcycles

*function*

```python
def wrap_subcycles(operations, plans, call, failure)
```

Retain per-merge field/cohort operations inside the guarded loop.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_subcycling.py#L347-L360)
