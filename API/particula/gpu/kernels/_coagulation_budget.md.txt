# particula.gpu.kernels._coagulation_budget

*module*

Explicit work admission for equal-count particle-resolved coagulation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L1-L1)

## particula.gpu.kernels._coagulation_budget.CoagulationBudgetError

*class*

```python
class CoagulationBudgetError(ValueError)
```

Bases: `ValueError`

An entire coagulation operation was rejected before mutable work.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L60-L61)

## particula.gpu.kernels._coagulation_budget.CoagulationWorkConfig

*class* · *dataclass*

```python
class CoagulationWorkConfig
```

Bound execution work independently of the physical population size.

**Attributes**

- `max_trials_per_box` (`int`) — Maximum scheduled candidate trials per step.
- `max_collision_buffer_bytes` (`int`) — Maximum collision-output allocation [byte].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L10-L32)

### particula.gpu.kernels._coagulation_budget.CoagulationWorkConfig.max_collision_buffer_bytes

*attribute* · *class attribute* · *instance attribute*

```python
max_collision_buffer_bytes: int = 256 * 1024 * 1024
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L20-L20)

### particula.gpu.kernels._coagulation_budget.CoagulationWorkConfig.max_trials_per_box

*attribute* · *class attribute* · *instance attribute*

```python
max_trials_per_box: int = 65536
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L19-L19)

### particula.gpu.kernels._coagulation_budget.CoagulationWorkConfig.__init__

*method*

```python
def __init__(max_trials_per_box: int = 65536, max_collision_buffer_bytes: int = 256 * 1024 * 1024) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L0-L0)

### particula.gpu.kernels._coagulation_budget.CoagulationWorkConfig.__post_init__

*method*

```python
def __post_init__()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L22-L32)

## particula.gpu.kernels._coagulation_budget.admit_coagulation_work

*function*

```python
def admit_coagulation_work(counts: wp.array(dtype=wp.int32), indices: wp.array2d(dtype=wp.int32), concentration: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64), majorants: wp.array(dtype=wp.float64), time_step: wp.float64, trial_budget: wp.int32, collision_capacity: wp.int32, report: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Report demand per box and latch rejection before any selector launch.

Report columns are expected trials, conservative trial bound, available
trials, and maximum budget-admissible timestep [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L64-L113)

## particula.gpu.kernels._coagulation_budget.initialize_admitted_rng

*function*

```python
def initialize_admitted_rng(rng: wp.array(dtype=wp.uint32), pending: wp.array(dtype=wp.int32), seed: wp.int32, failure: wp.array(dtype=wp.int32))
```

Initialize each stream once after successful admission.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L128-L141)

## particula.gpu.kernels._coagulation_budget.raise_budget_error

*function*

```python
def raise_budget_error(report)
```

Observe only compact admission diagnostics, never particle state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L116-L125)

## particula.gpu.kernels._coagulation_budget.validate_coagulation_layout

*function*

```python
def validate_coagulation_layout(boxes: int, slots: int, species: int) -> None
```

Reject unaddressable native layouts before allocating device scratch.

Warp dimensions and slot indices are signed int32. Pair ranks use int64;
host byte calculations use Python integers so multiplication cannot wrap.
Physical allocation failures remain explicit device allocation errors.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_coagulation_budget.py#L35-L57)
