# particula.execution.preparation_context

*module*

Typed, setup-only dependencies for built-in resident process adapters.

The context contains resources and preparation results, not simulation lifecycle
methods. Built-ins cannot step, close, snapshot or restart their owning runner.
Warp buffers remain runtime-typed at the direct-kernel boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L1-L1)

## particula.execution.preparation_context.BuiltinPreparationContext

*class* · *dataclass*

```python
class BuiltinPreparationContext
```

Explicit native resources and shared setup outputs for one preparation.

Binding and auxiliary maps are shared with the owner to preserve resource
identity. Only setup mutates these maps; retained operations capture their
fixed buffers. ``refresh`` publishes the derived-state refresh operation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L29-L64)

### particula.execution.preparation_context.BuiltinPreparationContext.auxiliary

*attribute* · *instance attribute*

```python
auxiliary: dict[str, Any]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L47-L47)

### particula.execution.preparation_context.BuiltinPreparationContext.bindings

*attribute* · *instance attribute*

```python
bindings: dict[str, Any]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L46-L46)

### particula.execution.preparation_context.BuiltinPreparationContext.device

*attribute* · *instance attribute*

```python
device: Device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L41-L41)

### particula.execution.preparation_context.BuiltinPreparationContext.failure

*attribute* · *instance attribute*

```python
failure: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L43-L43)

### particula.execution.preparation_context.BuiltinPreparationContext.fields

*attribute* · *instance attribute*

```python
fields: FieldBuffers
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L45-L45)

### particula.execution.preparation_context.BuiltinPreparationContext.observers

*attribute* · *instance attribute*

```python
observers: list[Callable[[], object]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L48-L48)

### particula.execution.preparation_context.BuiltinPreparationContext.plans

*attribute* · *instance attribute*

```python
plans: tuple[_ProcessPlan | _RegisteredPlan, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L44-L44)

### particula.execution.preparation_context.BuiltinPreparationContext.refresh

*attribute* · *class attribute* · *instance attribute*

```python
refresh: Callable[[], None] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L49-L49)

### particula.execution.preparation_context.BuiltinPreparationContext.registry

*attribute* · *instance attribute*

```python
registry: GPUResourceRegistry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L40-L40)

### particula.execution.preparation_context.BuiltinPreparationContext.session

*attribute* · *instance attribute*

```python
session: ResidentSession
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L39-L39)

### particula.execution.preparation_context.BuiltinPreparationContext.time_step

*attribute* · *instance attribute*

```python
time_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L42-L42)

### particula.execution.preparation_context.BuiltinPreparationContext.wp

*attribute* · *instance attribute*

```python
wp: ModuleType
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L38-L38)

### particula.execution.preparation_context.BuiltinPreparationContext.environment

*property*

```python
environment: WarpEnvironmentData
```

Expose the already validated box environment buffer layout.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L62-L64)

### particula.execution.preparation_context.BuiltinPreparationContext.gas

*property*

```python
gas: WarpGasData
```

Expose the already validated full-gas buffer layout.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L57-L59)

### particula.execution.preparation_context.BuiltinPreparationContext.particles

*property*

```python
particles: WarpParticleData
```

Expose the already validated native particle buffer layout.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L52-L54)

### particula.execution.preparation_context.BuiltinPreparationContext.__init__

*method*

```python
def __init__(wp: ModuleType, session: ResidentSession, registry: GPUResourceRegistry, device: Device, time_step: float, failure: Any, plans: tuple[_ProcessPlan | _RegisteredPlan, ...], fields: FieldBuffers, bindings: dict[str, Any], auxiliary: dict[str, Any], observers: list[Callable[[], object]], refresh: Callable[[], None] | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/preparation_context.py#L0-L0)
