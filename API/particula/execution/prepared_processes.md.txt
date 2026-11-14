# particula.execution.prepared_processes

*module*

Translate supported native runnables into immutable resident process plans.

Host validation finishes before any state upload. Compilers then allocate only
the selected process resources and retain the existing direct-kernel enqueue
delegates. User-defined strategies never receive an inferred GPU equivalent.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L1-L1)

## particula.execution.prepared_processes._ProcessPlan

*class* · *dataclass*

```python
class _ProcessPlan
```

Own normalized scalar/tuple configuration independent of CPU objects.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L18-L28)

### particula.execution.prepared_processes._ProcessPlan.family

*attribute* · *instance attribute*

```python
family: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L22-L22)

### particula.execution.prepared_processes._ProcessPlan.parameters

*attribute* · *class attribute* · *instance attribute*

```python
parameters: tuple[tuple[str, Any], ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L23-L23)

### particula.execution.prepared_processes._ProcessPlan.values

*property*

```python
values: dict[str, Any]
```

Make setup-only keyword arguments from the immutable plan.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L26-L28)

### particula.execution.prepared_processes._ProcessPlan.__init__

*method*

```python
def __init__(family: str, parameters: tuple[tuple[str, Any], ...] = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L0-L0)

## particula.execution.prepared_processes._compile_process

*function*

```python
def _compile_process(plan: _ProcessPlan, owner: Any)
```

Retain one existing prepared direct-kernel call and its resources.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L168-L273)

## particula.execution.prepared_processes._plan_coagulation

*function*

```python
def _plan_coagulation(process: Any, aerosol: Aerosol) -> _ProcessPlan
```

Admit direct Brownian physics and equal active represented counts.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L75-L98)

## particula.execution.prepared_processes._plan_dilution

*function*

```python
def _plan_dilution(process: Any, _aerosol: Aerosol) -> _ProcessPlan
```

Freeze the exact exponential dilution coefficient in inverse seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L59-L72)

## particula.execution.prepared_processes._plan_process

*function*

```python
def _plan_process(process: Any, aerosol: Aerosol) -> _ProcessPlan
```

Reject unknown runnable/strategy semantics before any GPU allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L31-L56)

## particula.execution.prepared_processes._plan_wall_loss

*function*

```python
def _plan_wall_loss(process: Any, aerosol: Aerosol) -> _ProcessPlan
```

Freeze neutral/charged chamber geometry and explicit electric fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L101-L165)
