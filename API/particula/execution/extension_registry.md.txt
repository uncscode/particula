# particula.execution.extension_registry

*module*

Explicit, simulation-local registration of process implementations.

Registration performs no runtime/device initialization. Exact type matching
prevents an adapter from silently accepting changed subclass semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L1-L1)

## particula.execution.extension_registry.AdapterT

*attribute* · *module attribute*

```python
AdapterT = TypeVar('AdapterT')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L13-L13)

## particula.execution.extension_registry.PreparedOperation

*class* · *dataclass*

```python
class PreparedOperation
```

Retain a named operation and the resources required for its lifetime.

The handler enqueues already prepared work. Resource descriptors remain
visible to capture identity validation rather than hidden in closures.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L89-L117)

### particula.execution.extension_registry.PreparedOperation.capabilities

*attribute* · *class attribute* · *instance attribute*

```python
capabilities: ProcessCapabilities = ProcessCapabilities()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L100-L100)

### particula.execution.extension_registry.PreparedOperation.handler

*attribute* · *instance attribute*

```python
handler: Callable[[], object]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L98-L98)

### particula.execution.extension_registry.PreparedOperation.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L97-L97)

### particula.execution.extension_registry.PreparedOperation.resources

*attribute* · *class attribute* · *instance attribute*

```python
resources: tuple[object, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L99-L99)

### particula.execution.extension_registry.PreparedOperation.__call__

*method*

```python
def __call__() -> None
```

Delegate exactly once without recovery, transfer or allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L115-L117)

### particula.execution.extension_registry.PreparedOperation.__init__

*method*

```python
def __init__(name: str, handler: Callable[[], object], resources: tuple[object, ...] = (), capabilities: ProcessCapabilities = ProcessCapabilities()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L0-L0)

### particula.execution.extension_registry.PreparedOperation.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate descriptors before any execution can occur.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L102-L113)

## particula.execution.extension_registry.ProcessCapabilities

*class* · *dataclass*

```python
class ProcessCapabilities
```

Declare independently supported extension execution boundaries.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L16-L30)

### particula.execution.extension_registry.ProcessCapabilities.capture

*attribute* · *class attribute* · *instance attribute*

```python
capture: bool = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L21-L21)

### particula.execution.extension_registry.ProcessCapabilities.checkpoint

*attribute* · *class attribute* · *instance attribute*

```python
checkpoint: bool = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L22-L22)

### particula.execution.extension_registry.ProcessCapabilities.resident

*attribute* · *class attribute* · *instance attribute*

```python
resident: bool = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L20-L20)

### particula.execution.extension_registry.ProcessCapabilities.__init__

*method*

```python
def __init__(resident: bool = False, capture: bool = False, checkpoint: bool = False) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L0-L0)

### particula.execution.extension_registry.ProcessCapabilities.__post_init__

*method*

```python
def __post_init__() -> None
```

Reject contradictory or ambiguous declarations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L24-L30)

## particula.execution.extension_registry.ProcessRegistry

*class*

```python
class ProcessRegistry(Generic[AdapterT])
```

Bases: ``

Own exact-type adapter registrations without global side effects.

A prepared simulation takes a snapshot to isolate its registrations from
subsequent edits to the caller's registry. Adapter configuration itself
must be frozen separately when creating a process plan.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L33-L86)

### particula.execution.extension_registry.ProcessRegistry._adapters

*attribute* · *instance attribute*

```python
_adapters: dict[type, AdapterT] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L43-L43)

### particula.execution.extension_registry.ProcessRegistry._frozen

*attribute* · *instance attribute*

```python
_frozen = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L44-L44)

### particula.execution.extension_registry.ProcessRegistry.__init__

*method*

```python
def __init__() -> None
```

Create an empty caller-owned registry.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L41-L44)

### particula.execution.extension_registry.ProcessRegistry.has_type

*method*

```python
def has_type(process_type: type) -> bool
```

Report exact registration without invoking user adapter code.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L68-L70)

### particula.execution.extension_registry.ProcessRegistry.register

*method*

```python
def register(process_type: type, adapter: AdapterT) -> None
```

Register one implementation, rejecting implicit replacement.

**Parameters**

- `process_type` (`type`) — Exact concrete process class.
- `adapter` (`AdapterT`) — Implementation retained by identity.

**Raises**

- `TypeError` — If the key is not a class or the adapter is None.
- `ValueError` — If the class already has a registration.
- `RuntimeError` — If this is an immutable preparation snapshot.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L46-L66)

### particula.execution.extension_registry.ProcessRegistry.resolve

*method*

```python
def resolve(process: object) -> AdapterT
```

Resolve the exact type without inheriting parent registrations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L72-L79)

### particula.execution.extension_registry.ProcessRegistry.snapshot

*method*

```python
def snapshot() -> ProcessRegistry[AdapterT]
```

Retain an immutable copy of the registration map.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_registry.py#L81-L86)
