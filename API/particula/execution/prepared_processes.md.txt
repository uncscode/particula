# particula.execution.prepared_processes

*module*

Translate supported native runnables into immutable resident process plans.

Host validation finishes before any state upload. Compilers then allocate only
the selected process resources and retain the existing direct-kernel enqueue
delegates. User-defined strategies never receive an inferred GPU equivalent.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L1-L1)

## particula.execution.prepared_processes._BuiltinAdapter

*class* · *dataclass*

```python
class _BuiltinAdapter
```

Pair admitted native planning with resource-only preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L54-L83)

### particula.execution.prepared_processes._BuiltinAdapter.compiler

*attribute* · *instance attribute*

```python
compiler: Callable[[_ProcessPlan, BuiltinPreparationContext], Callable[[], object]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L59-L61)

### particula.execution.prepared_processes._BuiltinAdapter.planner

*attribute* · *instance attribute*

```python
planner: Callable[[Any, Aerosol], _ProcessPlan]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L58-L58)

### particula.execution.prepared_processes._BuiltinAdapter.__init__

*method*

```python
def __init__(planner: Callable[[Any, Aerosol], _ProcessPlan], compiler: Callable[[_ProcessPlan, BuiltinPreparationContext], Callable[[], object]]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L0-L0)

### particula.execution.prepared_processes._BuiltinAdapter.plan

*method*

```python
def plan(process: Any, aerosol: Aerosol) -> _ProcessPlan
```

Keep exact native validation ahead of all GPU allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L63-L66)

### particula.execution.prepared_processes._BuiltinAdapter.prepare

*method*

```python
def prepare(plan: _ProcessPlan, context: BuiltinPreparationContext) -> PreparedOperation
```

Retain the launch handler and fixed native resources explicitly.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L68-L83)

## particula.execution.prepared_processes._ProcessPlan

*class* · *dataclass*

```python
class _ProcessPlan
```

Own normalized scalar/tuple configuration independent of CPU objects.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L25-L35)

### particula.execution.prepared_processes._ProcessPlan.family

*attribute* · *instance attribute*

```python
family: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L29-L29)

### particula.execution.prepared_processes._ProcessPlan.parameters

*attribute* · *class attribute* · *instance attribute*

```python
parameters: tuple[tuple[str, Any], ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L30-L30)

### particula.execution.prepared_processes._ProcessPlan.values

*property*

```python
values: dict[str, Any]
```

Make setup-only keyword arguments from the immutable plan.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L33-L35)

### particula.execution.prepared_processes._ProcessPlan.__init__

*method*

```python
def __init__(family: str, parameters: tuple[tuple[str, Any], ...] = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L0-L0)

## particula.execution.prepared_processes._builtin_adapters_by_family

*function*

```python
def _builtin_adapters_by_family() -> dict[str, _BuiltinAdapter]
```

Resolve normalized families without changing scientific admission.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L86-L120)

## particula.execution.prepared_processes._builtin_registry

*function*

```python
def _builtin_registry() -> ProcessRegistry[_BuiltinAdapter]
```

Build exact-type native adapters using the common registry contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L123-L180)

## particula.execution.prepared_processes._compile_coagulation

*function*

```python
def _compile_coagulation(plan, context: BuiltinPreparationContext)
```

Retain the common collision sampler with optional ion rate work.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L386-L438)

## particula.execution.prepared_processes._compile_dilution

*function*

```python
def _compile_dilution(plan: _ProcessPlan, context: BuiltinPreparationContext)
```

Bind the native exponential dilution operation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L321-L342)

## particula.execution.prepared_processes._compile_process

*function*

```python
def _compile_process(plan: _ProcessPlan, context: BuiltinPreparationContext)
```

Prepare a built-in through its explicit adapter contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L313-L318)

## particula.execution.prepared_processes._compile_wall_loss

*function*

```python
def _compile_wall_loss(plan: _ProcessPlan, context: BuiltinPreparationContext)
```

Bind the admitted neutral or charged wall-loss operation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L345-L383)

## particula.execution.prepared_processes._plan_coagulation

*function*

```python
def _plan_coagulation(process: Any, aerosol: Aerosol) -> _ProcessPlan
```

Admit direct Brownian physics and equal active represented counts.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L199-L243)

## particula.execution.prepared_processes._plan_dilution

*function*

```python
def _plan_dilution(process: Any, _aerosol: Aerosol) -> _ProcessPlan
```

Freeze the exact exponential dilution coefficient in inverse seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L183-L196)

## particula.execution.prepared_processes._plan_process

*function*

```python
def _plan_process(process: Any, aerosol: Aerosol) -> _ProcessPlan
```

Reject unknown runnable/strategy semantics before any GPU allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L38-L51)

## particula.execution.prepared_processes._plan_wall_loss

*function*

```python
def _plan_wall_loss(process: Any, aerosol: Aerosol) -> _ProcessPlan
```

Freeze neutral/charged chamber geometry and explicit electric fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_processes.py#L246-L310)
