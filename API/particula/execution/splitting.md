# particula.execution.splitting

*module*

Explicit deterministic process grouping and accepted complete-cycle clocks.

Each component advances its own physical interval. In a symmetric group,
outer components advance [t,t+h/2] then [t+h/2,t+h], around the inner full step.
Groups are never flattened. Observers belong outside this physics scheduler.
Stochastic updates and equilibrium projections require separate contracts and
are rejected here rather than being silently passed through symmetric stages.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L1-L1)

## particula.execution.splitting.Flow

*class* · *dataclass*

```python
class Flow
```

Declare an exact map or deterministic continuous component.

advance(state, start_time, duration) receives detached native state and may
mutate/return it. It must not mutate external resources or consume RNG.
order declares the component's convergence order, not a scheduler guarantee.
Exact finite-step maps use kind='exact'; continuous methods use
kind='continuous'. Neither runnable.rate nor execute is inferred here.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L20-L47)

### particula.execution.splitting.Flow.advance

*attribute* · *instance attribute*

```python
advance: Callable[[Aerosol, float, float], Aerosol]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L32-L32)

### particula.execution.splitting.Flow.kind

*attribute* · *class attribute* · *instance attribute*

```python
kind: str = 'continuous'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L33-L33)

### particula.execution.splitting.Flow.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L31-L31)

### particula.execution.splitting.Flow.order

*attribute* · *class attribute* · *instance attribute*

```python
order: int = 2
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L34-L34)

### particula.execution.splitting.Flow.__init__

*method*

```python
def __init__(name: str, advance: Callable[[Aerosol, float, float], Aerosol], kind: str = 'continuous', order: int = 2) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L0-L0)

### particula.execution.splitting.Flow.__post_init__

*method*

```python
def __post_init__()
```

Reject ambiguous names and unsupported process contracts early.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L36-L47)

## particula.execution.splitting.SplitGroup

*class* · *dataclass*

```python
class SplitGroup
```

Retain nesting and sequential or symmetric composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L50-L74)

### particula.execution.splitting.SplitGroup.children

*attribute* · *instance attribute*

```python
children: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L54-L54)

### particula.execution.splitting.SplitGroup.method

*attribute* · *class attribute* · *instance attribute*

```python
method: str = 'strang'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L55-L55)

### particula.execution.splitting.SplitGroup.__init__

*method*

```python
def __init__(children: tuple, method: str = 'strang') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L0-L0)

### particula.execution.splitting.SplitGroup.__post_init__

*method*

```python
def __post_init__()
```

Freeze groups and validate component accuracy declarations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L57-L74)

## particula.execution.splitting.SplitSimulation

*class*

```python
class SplitSimulation
```

Own native accepted state and publish only complete deterministic cycles.

A component failure discards the full current cycle; earlier accepted cycles
remain. Callbacks must obey Flow's pure-resource contract. Fixed outer steps
limit splitting error independently of adaptive component error control.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L111-L163)

### particula.execution.splitting.SplitSimulation._state

*attribute* · *instance attribute*

```python
_state = _cpu_batch.detached(aerosol)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L124-L124)

### particula.execution.splitting.SplitSimulation._time

*attribute* · *instance attribute*

```python
_time = _cpu_batch.duration(time, 1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L127-L127)

### particula.execution.splitting.SplitSimulation.completed_steps

*attribute* · *instance attribute*

```python
completed_steps = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L130-L130)

### particula.execution.splitting.SplitSimulation.group

*attribute* · *instance attribute*

```python
group = group
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L125-L125)

### particula.execution.splitting.SplitSimulation.time_step

*attribute* · *instance attribute*

```python
time_step = _cpu_batch.duration(time_step, 1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L126-L126)

### particula.execution.splitting.SplitSimulation.time

*property*

```python
time
```

Return accepted absolute time [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L133-L135)

### particula.execution.splitting.SplitSimulation.__init__

*method*

```python
def __init__(aerosol, group, *, time_step, time=0.0)
```

Validate schedule and detach the initial accepted state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L119-L130)

### particula.execution.splitting.SplitSimulation.snapshot

*method*

```python
def snapshot()
```

Return detached state after a complete accepted cycle.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L161-L163)

### particula.execution.splitting.SplitSimulation.step

*method*

```python
def step(bound)
```

Advance one complete cycle, shortened to land on bound [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L137-L159)

## particula.execution.splitting._advance

*function*

```python
def _advance(node, state, time, duration)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L94-L108)

## particula.execution.splitting._leaves

*function*

```python
def _leaves(node)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L86-L91)

## particula.execution.splitting._order

*function*

```python
def _order(node)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L77-L83)

## particula.execution.splitting.condensation_flow

*function*

```python
def condensation_flow(name, strategy, **controls)
```

Construct continuous native condensation with explicit controls.

Component controllers restart at split boundaries. The outer step and
component tolerances must both be refined to establish trajectory accuracy.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L166-L184)

## particula.execution.splitting.dilution_flow

*function*

```python
def dilution_flow(name, coefficient)
```

Construct exact native gas/particle dilution with hazard [s^-1].

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/splitting.py#L187-L199)
