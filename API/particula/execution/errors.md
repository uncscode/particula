# particula.execution.errors

*module*

Dependency-neutral execution capability error taxonomy.

This module defines direct-import-only errors for execution availability and
fallback boundaries. It intentionally imports no execution providers or
optional runtimes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L1-L1)

## particula.execution.errors.ExecutionCapabilityError

*class*

```python
class ExecutionCapabilityError(Exception)
```

Bases: `Exception`

Describe an unavailable, unsupported, or invalid execution request.

**Parameters**

- `reason` (`ExecutionCapabilityReason`) — Stable reason code for the error.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.
- `fallback_boundary` (`str | None`) (default: `None`) — Optional fallback-boundary context.

**Raises**

- `TypeError` — If a reason or context field has an unsupported type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L24-L109)

### particula.execution.errors.ExecutionCapabilityError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.ExecutionCapabilityError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.ExecutionCapabilityError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.ExecutionCapabilityError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.ExecutionCapabilityError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.ExecutionCapabilityError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.ExecutionCapabilityError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.ExecutionCapabilityError.__init__

*method*

```python
def __init__(reason: ExecutionCapabilityReason, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None, state: str | None = None, fallback_boundary: str | None = None) -> None
```

Initialize the structured capability error.

**Parameters**

- `reason` (`ExecutionCapabilityReason`) — Stable reason code for the error.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.
- `fallback_boundary` (`str | None`) (default: `None`) — Optional fallback-boundary context.

**Raises**

- `TypeError` — If a reason or context field has an unsupported type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L48-L95)

### particula.execution.errors.ExecutionCapabilityError.__str__

*method*

```python
def __str__() -> str
```

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.ExecutionCapabilityReason

*class*

```python
class ExecutionCapabilityReason(str, Enum)
```

Bases: `str`, `Enum`

Closed reason codes for execution capability errors.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L11-L21)

### particula.execution.errors.ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED

*attribute* · *class attribute* · *instance attribute*

```python
CAPABILITY_UNSUPPORTED = 'capability_unsupported'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L19-L19)

### particula.execution.errors.ExecutionCapabilityReason.DEVICE_UNAVAILABLE

*attribute* · *class attribute* · *instance attribute*

```python
DEVICE_UNAVAILABLE = 'device_unavailable'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L17-L17)

### particula.execution.errors.ExecutionCapabilityReason.FALLBACK_DISALLOWED

*attribute* · *class attribute* · *instance attribute*

```python
FALLBACK_DISALLOWED = 'fallback_disallowed'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L21-L21)

### particula.execution.errors.ExecutionCapabilityReason.INVALID_STATE

*attribute* · *class attribute* · *instance attribute*

```python
INVALID_STATE = 'invalid_state'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L20-L20)

### particula.execution.errors.ExecutionCapabilityReason.PROCESS_UNSUPPORTED

*attribute* · *class attribute* · *instance attribute*

```python
PROCESS_UNSUPPORTED = 'process_unsupported'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L18-L18)

### particula.execution.errors.ExecutionCapabilityReason.RUNTIME_UNAVAILABLE

*attribute* · *class attribute* · *instance attribute*

```python
RUNTIME_UNAVAILABLE = 'runtime_unavailable'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L16-L16)

### particula.execution.errors.ExecutionCapabilityReason.UNKNOWN_BACKEND

*attribute* · *class attribute* · *instance attribute*

```python
UNKNOWN_BACKEND = 'unknown_backend'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L14-L14)

### particula.execution.errors.ExecutionCapabilityReason.UNKNOWN_DEVICE

*attribute* · *class attribute* · *instance attribute*

```python
UNKNOWN_DEVICE = 'unknown_device'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L15-L15)

## particula.execution.errors.FallbackDisallowedError

*class*

```python
class FallbackDisallowedError(ExecutionCapabilityError)
```

Bases: `particula.execution.errors.ExecutionCapabilityError`

Report a fallback request outside its permitted boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L272-L303)

### particula.execution.errors.FallbackDisallowedError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.FallbackDisallowedError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.FallbackDisallowedError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.FallbackDisallowedError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.FallbackDisallowedError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.FallbackDisallowedError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.FallbackDisallowedError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.FallbackDisallowedError.__init__

*method*

```python
def __init__(fallback_boundary: str, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None, state: str | None = None) -> None
```

Initialize a fallback-disallowed error.

**Parameters**

- `fallback_boundary` (`str`) — Boundary that disallows the fallback request.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L275-L303)

### particula.execution.errors.FallbackDisallowedError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.InvalidExecutionStateError

*class*

```python
class InvalidExecutionStateError(ExecutionCapabilityError)
```

Bases: `particula.execution.errors.ExecutionCapabilityError`

Report an invalid state for an execution request.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L241-L269)

### particula.execution.errors.InvalidExecutionStateError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.InvalidExecutionStateError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.InvalidExecutionStateError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.InvalidExecutionStateError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.InvalidExecutionStateError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.InvalidExecutionStateError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.InvalidExecutionStateError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.InvalidExecutionStateError.__init__

*method*

```python
def __init__(state: str, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None) -> None
```

Initialize an invalid-execution-state error.

**Parameters**

- `state` (`str`) — Execution state that is invalid for the request.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L244-L269)

### particula.execution.errors.InvalidExecutionStateError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.UnavailableDeviceError

*class*

```python
class UnavailableDeviceError(UnavailableExecutionTargetError)
```

Bases: `particula.execution.errors.UnavailableExecutionTargetError`

Report a device that is unavailable for execution.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L171-L185)

### particula.execution.errors.UnavailableDeviceError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.UnavailableDeviceError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.UnavailableDeviceError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.UnavailableDeviceError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.UnavailableDeviceError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.UnavailableDeviceError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.UnavailableDeviceError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.UnavailableDeviceError.__init__

*method*

```python
def __init__(device: str, *, backend: str | None = None) -> None
```

Initialize an unavailable-device error.

**Parameters**

- `device` (`str`) — Device that is unavailable for execution.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L174-L185)

### particula.execution.errors.UnavailableDeviceError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.UnavailableExecutionTargetError

*class*

```python
class UnavailableExecutionTargetError(ExecutionCapabilityError)
```

Bases: `particula.execution.errors.ExecutionCapabilityError`

Base category for unavailable execution targets.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L116-L117)

### particula.execution.errors.UnavailableExecutionTargetError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.UnavailableExecutionTargetError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.UnavailableExecutionTargetError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.UnavailableExecutionTargetError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.UnavailableExecutionTargetError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.UnavailableExecutionTargetError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.UnavailableExecutionTargetError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.UnavailableExecutionTargetError.__init__

*method*

```python
def __init__(reason: ExecutionCapabilityReason, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None, state: str | None = None, fallback_boundary: str | None = None) -> None
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Initialize the structured capability error.

**Parameters**

- `reason` (`ExecutionCapabilityReason`) — Stable reason code for the error.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.
- `fallback_boundary` (`str | None`) (default: `None`) — Optional fallback-boundary context.

**Raises**

- `TypeError` — If a reason or context field has an unsupported type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L48-L95)

### particula.execution.errors.UnavailableExecutionTargetError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.UnavailableRuntimeError

*class*

```python
class UnavailableRuntimeError(UnavailableExecutionTargetError)
```

Bases: `particula.execution.errors.UnavailableExecutionTargetError`

Report an execution backend whose runtime is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L156-L168)

### particula.execution.errors.UnavailableRuntimeError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.UnavailableRuntimeError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.UnavailableRuntimeError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.UnavailableRuntimeError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.UnavailableRuntimeError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.UnavailableRuntimeError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.UnavailableRuntimeError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.UnavailableRuntimeError.__init__

*method*

```python
def __init__(backend: str) -> None
```

Initialize an unavailable-runtime error.

**Parameters**

- `backend` (`str`) — Backend whose required runtime is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L159-L168)

### particula.execution.errors.UnavailableRuntimeError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.UnknownBackendError

*class*

```python
class UnknownBackendError(UnknownExecutionTargetError)
```

Bases: `particula.execution.errors.UnknownExecutionTargetError`

Report a backend identifier that is not recognized.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L124-L136)

### particula.execution.errors.UnknownBackendError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.UnknownBackendError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.UnknownBackendError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.UnknownBackendError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.UnknownBackendError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.UnknownBackendError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.UnknownBackendError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.UnknownBackendError.__init__

*method*

```python
def __init__(backend: str) -> None
```

Initialize an unknown-backend error.

**Parameters**

- `backend` (`str`) — Unrecognized execution backend identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L127-L136)

### particula.execution.errors.UnknownBackendError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.UnknownDeviceError

*class*

```python
class UnknownDeviceError(UnknownExecutionTargetError)
```

Bases: `particula.execution.errors.UnknownExecutionTargetError`

Report a device identifier that is not recognized.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L139-L153)

### particula.execution.errors.UnknownDeviceError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.UnknownDeviceError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.UnknownDeviceError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.UnknownDeviceError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.UnknownDeviceError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.UnknownDeviceError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.UnknownDeviceError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.UnknownDeviceError.__init__

*method*

```python
def __init__(device: str, *, backend: str | None = None) -> None
```

Initialize an unknown-device error.

**Parameters**

- `device` (`str`) — Unrecognized execution device identifier.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L142-L153)

### particula.execution.errors.UnknownDeviceError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.UnknownExecutionTargetError

*class*

```python
class UnknownExecutionTargetError(ExecutionCapabilityError)
```

Bases: `particula.execution.errors.ExecutionCapabilityError`

Base category for unknown execution targets.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L112-L113)

### particula.execution.errors.UnknownExecutionTargetError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.UnknownExecutionTargetError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.UnknownExecutionTargetError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.UnknownExecutionTargetError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.UnknownExecutionTargetError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.UnknownExecutionTargetError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.UnknownExecutionTargetError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.UnknownExecutionTargetError.__init__

*method*

```python
def __init__(reason: ExecutionCapabilityReason, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None, state: str | None = None, fallback_boundary: str | None = None) -> None
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Initialize the structured capability error.

**Parameters**

- `reason` (`ExecutionCapabilityReason`) — Stable reason code for the error.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.
- `fallback_boundary` (`str | None`) (default: `None`) — Optional fallback-boundary context.

**Raises**

- `TypeError` — If a reason or context field has an unsupported type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L48-L95)

### particula.execution.errors.UnknownExecutionTargetError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.UnsupportedCapabilityError

*class*

```python
class UnsupportedCapabilityError(UnsupportedExecutionRequestError)
```

Bases: `particula.execution.errors.UnsupportedExecutionRequestError`

Report an execution capability that is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L213-L238)

### particula.execution.errors.UnsupportedCapabilityError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.UnsupportedCapabilityError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.UnsupportedCapabilityError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.UnsupportedCapabilityError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.UnsupportedCapabilityError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.UnsupportedCapabilityError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.UnsupportedCapabilityError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.UnsupportedCapabilityError.__init__

*method*

```python
def __init__(capability: str, *, backend: str | None = None, device: str | None = None, process: str | None = None) -> None
```

Initialize an unsupported-capability error.

**Parameters**

- `capability` (`str`) — Execution capability that is not supported.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L216-L238)

### particula.execution.errors.UnsupportedCapabilityError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.UnsupportedExecutionRequestError

*class*

```python
class UnsupportedExecutionRequestError(ExecutionCapabilityError)
```

Bases: `particula.execution.errors.ExecutionCapabilityError`

Base category for unsupported execution requests.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L120-L121)

### particula.execution.errors.UnsupportedExecutionRequestError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.UnsupportedExecutionRequestError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.UnsupportedExecutionRequestError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.UnsupportedExecutionRequestError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.UnsupportedExecutionRequestError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.UnsupportedExecutionRequestError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.UnsupportedExecutionRequestError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.UnsupportedExecutionRequestError.__init__

*method*

```python
def __init__(reason: ExecutionCapabilityReason, *, backend: str | None = None, device: str | None = None, process: str | None = None, capability: str | None = None, state: str | None = None, fallback_boundary: str | None = None) -> None
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Initialize the structured capability error.

**Parameters**

- `reason` (`ExecutionCapabilityReason`) — Stable reason code for the error.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.
- `process` (`str | None`) (default: `None`) — Optional process context.
- `capability` (`str | None`) (default: `None`) — Optional capability context.
- `state` (`str | None`) (default: `None`) — Optional execution-state context.
- `fallback_boundary` (`str | None`) (default: `None`) — Optional fallback-boundary context.

**Raises**

- `TypeError` — If a reason or context field has an unsupported type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L48-L95)

### particula.execution.errors.UnsupportedExecutionRequestError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)

## particula.execution.errors.UnsupportedProcessError

*class*

```python
class UnsupportedProcessError(UnsupportedExecutionRequestError)
```

Bases: `particula.execution.errors.UnsupportedExecutionRequestError`

Report an execution process that is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L188-L210)

### particula.execution.errors.UnsupportedProcessError.backend

*attribute* · *instance attribute*

```python
backend: str | None = backend
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L89-L89)

### particula.execution.errors.UnsupportedProcessError.capability

*attribute* · *instance attribute*

```python
capability: str | None = capability
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L92-L92)

### particula.execution.errors.UnsupportedProcessError.device

*attribute* · *instance attribute*

```python
device: str | None = device
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L90-L90)

### particula.execution.errors.UnsupportedProcessError.fallback_boundary

*attribute* · *instance attribute*

```python
fallback_boundary: str | None = fallback_boundary
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L94-L94)

### particula.execution.errors.UnsupportedProcessError.process

*attribute* · *instance attribute*

```python
process: str | None = process
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L91-L91)

### particula.execution.errors.UnsupportedProcessError.reason

*attribute* · *instance attribute*

```python
reason: ExecutionCapabilityReason = reason
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L88-L88)

### particula.execution.errors.UnsupportedProcessError.state

*attribute* · *instance attribute*

```python
state: str | None = state
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L93-L93)

### particula.execution.errors.UnsupportedProcessError.__init__

*method*

```python
def __init__(process: str, *, backend: str | None = None, device: str | None = None) -> None
```

Initialize an unsupported-process error.

**Parameters**

- `process` (`str`) — Execution process that is not supported.
- `backend` (`str | None`) (default: `None`) — Optional execution backend context.
- `device` (`str | None`) (default: `None`) — Optional execution device context.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L191-L210)

### particula.execution.errors.UnsupportedProcessError.__str__

*method*

```python
def __str__() -> str
```

Inherited from: `particula.execution.errors.ExecutionCapabilityError`

Render deterministic structured error context.

**Returns**

- (`str`) — Error class name, reason code, and ordered context fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/errors.py#L97-L109)
