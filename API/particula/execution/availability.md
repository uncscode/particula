# particula.execution.availability

*module*

Resolve declared execution availability without selecting execution work.

This concrete, direct-import-only boundary validates declaration request
metadata,
capability declarations, optional runtime and device status, and
request-associated state in a fixed order. It neither selects adapters nor
allocates execution resources, transfers, synchronizes, mutates, or executes
work.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L1-L1)

## particula.execution.availability.StateValidator

*attribute* · *module attribute*

```python
StateValidator = Callable[[ExecutionRequest], bool]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L72-L72)

## particula.execution.availability._DEFAULT_PROVIDERS

*attribute* · *module attribute*

```python
_DEFAULT_PROVIDERS: Mapping[Backend, AvailabilityProvider] = {Backend.CPU: _CPUAvailabilityProvider(), Backend.WARP: _WarpAvailabilityProvider()}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L175-L178)

## particula.execution.availability.AvailabilityDecision

*class* · *dataclass*

```python
class AvailabilityDecision
```

Record an availability-validated request without runtime ownership.

**Attributes**

- `request` (`ExecutionRequest`) — The exact validated request retained by identity. This record owns no adapter, runtime handle, device object, or execution state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L75-L84)

### particula.execution.availability.AvailabilityDecision.request

*attribute* · *instance attribute*

```python
request: ExecutionRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L84-L84)

### particula.execution.availability.AvailabilityDecision.__init__

*method*

```python
def __init__(request: ExecutionRequest) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L0-L0)

## particula.execution.availability.AvailabilityProvider

*class*

```python
class AvailabilityProvider(Protocol)
```

Bases: `Protocol`

Describe pure recognition and lazy status checks for one backend.

Implementations must not execute work. Runtime and device checks run only
after the resolver has accepted the request declaration and capabilities.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L37-L69)

### particula.execution.availability.AvailabilityProvider.device_available

*method*

```python
def device_available(device: Device) -> bool
```

Return whether a recognized device is available at runtime.

**Parameters**

- `device` (`Device`) — Structurally recognized device declaration to check.

**Returns**

- (`bool`) — ``True`` when the runtime can resolve the declared device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L61-L69)

### particula.execution.availability.AvailabilityProvider.recognizes

*method*

```python
def recognizes(device: Device) -> bool
```

Return whether a declared device is structurally recognized.

**Parameters**

- `device` (`Device`) — Validated device declaration to recognize without probing.

**Returns**

- (`bool`) — ``True`` when the provider recognizes the declaration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L44-L52)

### particula.execution.availability.AvailabilityProvider.runtime_available

*method*

```python
def runtime_available() -> bool
```

Return whether the optional backend runtime is available.

**Returns**

- (`bool`) — ``True`` when the backend runtime is available for device checking.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L54-L59)

## particula.execution.availability._CPUAvailabilityProvider

*class*

```python
class _CPUAvailabilityProvider
```

Provide no-probe availability checks for the canonical CPU device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L87-L119)

### particula.execution.availability._CPUAvailabilityProvider.device_available

*method*

```python
def device_available(device: Device) -> bool
```

Return CPU device availability without probing hardware.

**Parameters**

- `device` (`Device`) — Recognized canonical CPU declaration, unused by this check.

**Returns**

- (`bool`) — ``True`` without inspecting hardware.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L109-L119)

### particula.execution.availability._CPUAvailabilityProvider.recognizes

*method*

```python
def recognizes(device: Device) -> bool
```

Return whether the device is the canonical CPU declaration.

**Parameters**

- `device` (`Device`) — Validated device declaration to compare.

**Returns**

- (`bool`) — ``True`` only for ``Device(Backend.CPU, "cpu")``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L90-L99)

### particula.execution.availability._CPUAvailabilityProvider.runtime_available

*method*

```python
def runtime_available() -> bool
```

Return CPU runtime availability without probing hardware.

**Returns**

- (`bool`) — ``True``, because the CPU provider has no optional runtime.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L101-L107)

## particula.execution.availability._WarpAvailabilityProvider

*class*

```python
class _WarpAvailabilityProvider
```

Provide lazy runtime and device checks for opaque Warp devices.

Native identifiers are passed unchanged to Warp after lazy runtime loading;
this provider neither parses nor normalizes them.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L122-L172)

### particula.execution.availability._WarpAvailabilityProvider.device_available

*method*

```python
def device_available(device: Device) -> bool
```

Return whether Warp resolves the opaque native device identifier.

**Parameters**

- `device` (`Device`) — Recognized Warp declaration whose native value is opaque.

**Returns**

- (`bool`) — ``True`` when Warp resolves the original native identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L152-L172)

### particula.execution.availability._WarpAvailabilityProvider.recognizes

*method*

```python
def recognizes(device: Device) -> bool
```

Return whether the device declares the Warp backend.

**Parameters**

- `device` (`Device`) — Validated device declaration with an opaque native value.

**Returns**

- (`bool`) — ``True`` when the declaration uses ``Backend.WARP``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L129-L138)

### particula.execution.availability._WarpAvailabilityProvider.runtime_available

*method*

```python
def runtime_available() -> bool
```

Return whether Warp can be imported lazily.

**Returns**

- (`bool`) — ``True`` when the optional Warp runtime imports successfully.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L140-L150)

## particula.execution.availability._always_valid

*function*

```python
def _always_valid(_: ExecutionRequest) -> bool
```

Return the default successful request-associated state validation.

**Parameters**

- `_` (`ExecutionRequest`) — Validated request intentionally ignored by the default validator.

**Returns**

- (`bool`) — ``True`` for every valid request.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L181-L190)

## particula.execution.availability._context

*function*

```python
def _context(request: ExecutionRequest) -> dict[str, str]
```

Build structured error context from a validated request.

**Parameters**

- `request` (`ExecutionRequest`) — Validated request supplying backend, device, process, and capability metadata.

**Returns**

- (`dict[str, str]`) — Error-context fields for availability failures.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L193-L208)

## particula.execution.availability._provider_result

*function*

```python
def _provider_result(method: Callable[..., object], error_factory: Callable[[], Exception], *args: object) -> bool
```

Call a provider phase and map exceptions or malformed results.

**Parameters**

- `method` (`Callable[..., object]`) — Provider status method for one availability phase.
- `error_factory` (`Callable[[], Exception]`) — Creates the typed error for a failed phase.
- `*args` (`object`) (default: `()`) — Positional arguments forwarded to ``method``.

**Returns**

- (`bool`) — The validated boolean status returned by ``method``.

**Raises**

- `Exception` — The phase-specific error when the method raises or returns
a value other than an exact boolean.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L275-L300)

## particula.execution.availability._validate_providers

*function*

```python
def _validate_providers(providers: object, request: ExecutionRequest) -> tuple[Callable[[Device], object], Callable[[], object], Callable[[Device], object]]
```

Return a selected provider after fail-closed registry validation.

**Parameters**

- `providers` (`object`) — Candidate registry expected to contain exactly CPU and Warp providers with callable availability methods.
- `request` (`ExecutionRequest`) — Validated request selecting one provider.

**Returns**

- (`tuple[Callable[[Device], object], Callable[[], object], Callable[[Device], object]]`) — The validated bound provider methods for the request backend.

**Raises**

- `UnavailableRuntimeError` — If the registry is malformed or inaccessible.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L211-L272)

## particula.execution.availability.resolve_availability

*function*

```python
def resolve_availability(request: ExecutionRequest, matrix: CapabilityMatrix, *, providers: Mapping[Backend, AvailabilityProvider] | None = None, state_validator: StateValidator | None = None) -> AvailabilityDecision
```

Resolve availability in declaration, runtime, device, and state order.

The resolver is pre-execution only. It validates a provider registry before
provider calls, then short-circuits through recognition, capability
declarations, runtime status, device status, and request-associated state.

**Parameters**

- `request` (`ExecutionRequest`) — Validated declaration request declaration.
- `matrix` (`CapabilityMatrix`) — Capability declarations applicable to the exact device.
- `providers` (`Mapping[Backend, AvailabilityProvider] | None`) (default: `None`) — Optional complete CPU/Warp availability registry.
- `state_validator` (`StateValidator | None`) (default: `None`) — Optional request-associated state validator. Omitting it selects an internal validator that accepts every request, so state validation is substantive only when a validator is supplied.

**Returns**

- (`AvailabilityDecision`) — An immutable decision retaining the exact request by identity.

**Raises**

- `TypeError` — If a declaration carrier has an invalid type.
- `UnknownDeviceError` — If the selected provider rejects the device.
- `UnsupportedProcessError` — If no declaration matches the device/process.
- `UnsupportedCapabilityError` — If no declaration satisfies requirements.
- `UnavailableRuntimeError` — If a registry or runtime is unavailable.
- `UnavailableDeviceError` — If the runtime cannot resolve the device.
- `InvalidExecutionStateError` — If request-associated state is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/availability.py#L303-L403)
