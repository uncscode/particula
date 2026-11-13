# particula.execution.fallback

*module*

Provide a direct-import-only, explicit CPU fallback boundary.

This module does not import optional runtimes or perform transfers, lifecycle
operations, restoration, recovery, or retries. Callers retain responsibility
for providing CPU-authoritative state at a visible boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L1-L1)

## particula.execution.fallback._ELIGIBLE_REASONS

*attribute* · *module attribute*

```python
_ELIGIBLE_REASONS = frozenset({ExecutionCapabilityReason.UNKNOWN_DEVICE, ExecutionCapabilityReason.RUNTIME_UNAVAILABLE, ExecutionCapabilityReason.DEVICE_UNAVAILABLE, ExecutionCapabilityReason.PROCESS_UNSUPPORTED, ExecutionCapabilityReason.CAPABILITY_UNSUPPORTED})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L179-L187)

## particula.execution.fallback.FallbackDispatchResult

*class* · *dataclass*

```python
class FallbackDispatchResult
```

Retain a native result and immutable fallback provenance by identity.

**Attributes**

- `resolution` (`FallbackResolution`) — Identity-preserved CPU fallback resolution.
- `result` (`ExecutionResult`) — Identity-preserved, validated native adapter result.
- `metadata` (`tuple[tuple[str, str], ...]`) — Ordered fallback provenance separate from native metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L144-L176)

### particula.execution.fallback.FallbackDispatchResult.metadata

*attribute* · *instance attribute*

```python
metadata: tuple[tuple[str, str], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L156-L156)

### particula.execution.fallback.FallbackDispatchResult.resolution

*attribute* · *instance attribute*

```python
resolution: FallbackResolution
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L154-L154)

### particula.execution.fallback.FallbackDispatchResult.result

*attribute* · *instance attribute*

```python
result: ExecutionResult
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L155-L155)

### particula.execution.fallback.FallbackDispatchResult.__init__

*method*

```python
def __init__(resolution: FallbackResolution, result: ExecutionResult, metadata: tuple[tuple[str, str], ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L0-L0)

### particula.execution.fallback.FallbackDispatchResult.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate exact fallback metadata without changing native metadata.

**Raises**

- `TypeError` — If a carrier, native result, or metadata tuple is
invalid.
- `ValueError` — If native result validation or fallback metadata
validation fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L158-L176)

## particula.execution.fallback.FallbackRequest

*class* · *dataclass*

```python
class FallbackRequest
```

Carry one explicit fallback request without inspecting state payloads.

**Parameters**

- `original_request` (`ExecutionRequest`) — Original non-CPU selection request.
- `original_error` (`ExecutionCapabilityError`) — Typed capability error produced for that request.
- `context` (`ExecutionContext`) — Context owning the registered CPU adapter.
- `cpu_state` (`CPUExecutionState | None`) — Caller-owned CPU state, if fallback is enabled.
- `policy` (`FallbackPolicy`) (default: `FallbackPolicy.RAISE`) — Default-deny or explicit CPU fallback policy.
- `boundary` (`FallbackBoundary`) (default: `FallbackBoundary.PRE_UPLOAD`) — Caller-visible fallback boundary.
- `state_authority` (`CPUStateAuthority | None`) (default: `None`) — Caller assertion about CPU-state authority.

The carrier validates types only. It does not inspect opaque state payloads
or select an adapter.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L38-L84)

### particula.execution.fallback.FallbackRequest.boundary

*attribute* · *class attribute* · *instance attribute*

```python
boundary: FallbackBoundary = FallbackBoundary.PRE_UPLOAD
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L60-L60)

### particula.execution.fallback.FallbackRequest.context

*attribute* · *instance attribute*

```python
context: ExecutionContext
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L57-L57)

### particula.execution.fallback.FallbackRequest.cpu_state

*attribute* · *instance attribute*

```python
cpu_state: CPUExecutionState | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L58-L58)

### particula.execution.fallback.FallbackRequest.original_error

*attribute* · *instance attribute*

```python
original_error: ExecutionCapabilityError
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L56-L56)

### particula.execution.fallback.FallbackRequest.original_request

*attribute* · *instance attribute*

```python
original_request: ExecutionRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L55-L55)

### particula.execution.fallback.FallbackRequest.policy

*attribute* · *class attribute* · *instance attribute*

```python
policy: FallbackPolicy = FallbackPolicy.RAISE
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L59-L59)

### particula.execution.fallback.FallbackRequest.state_authority

*attribute* · *class attribute* · *instance attribute*

```python
state_authority: CPUStateAuthority | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L61-L61)

### particula.execution.fallback.FallbackRequest.__init__

*method*

```python
def __init__(original_request: ExecutionRequest, original_error: ExecutionCapabilityError, context: ExecutionContext, cpu_state: CPUExecutionState | None, policy: FallbackPolicy = FallbackPolicy.RAISE, boundary: FallbackBoundary = FallbackBoundary.PRE_UPLOAD, state_authority: CPUStateAuthority | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L0-L0)

### particula.execution.fallback.FallbackRequest.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate carrier types without selecting or reading payloads.

**Raises**

- `TypeError` — If a carrier field or fallback enum has an invalid type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L63-L84)

## particula.execution.fallback.FallbackResolution

*class* · *dataclass*

```python
class FallbackResolution
```

Retain the one CPU selection made for an eligible fallback request.

**Attributes**

- `original_request` (`ExecutionRequest`) — Identity-preserved non-CPU request.
- `cpu_request` (`ExecutionRequest`) — Canonical CPU request used for the single lookup.
- `original_error` (`ExecutionCapabilityError`) — Identity-preserved eligible capability error.
- `cpu_state` (`CPUExecutionState`) — Identity-preserved CPU state passed to the adapter.
- `adapter` (`ExecutionAdapter`) — Adapter returned by the single CPU lookup.
- `policy` (`FallbackPolicy`) — Explicit policy authorizing this resolution.
- `boundary` (`FallbackBoundary`) — Caller-visible CPU authority boundary.
- `requested_backend` (`Backend`) — Backend requested by the original request.
- `selected_backend` (`Backend`) — Canonical selected CPU backend.
- `capability_reason` (`ExecutionCapabilityReason`) — Capability reason retained from the original error.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L87-L141)

### particula.execution.fallback.FallbackResolution.adapter

*attribute* · *instance attribute*

```python
adapter: ExecutionAdapter
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L108-L108)

### particula.execution.fallback.FallbackResolution.boundary

*attribute* · *instance attribute*

```python
boundary: FallbackBoundary
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L110-L110)

### particula.execution.fallback.FallbackResolution.capability_reason

*attribute* · *instance attribute*

```python
capability_reason: ExecutionCapabilityReason
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L113-L113)

### particula.execution.fallback.FallbackResolution.cpu_request

*attribute* · *instance attribute*

```python
cpu_request: ExecutionRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L105-L105)

### particula.execution.fallback.FallbackResolution.cpu_state

*attribute* · *instance attribute*

```python
cpu_state: CPUExecutionState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L107-L107)

### particula.execution.fallback.FallbackResolution.original_error

*attribute* · *instance attribute*

```python
original_error: ExecutionCapabilityError
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L106-L106)

### particula.execution.fallback.FallbackResolution.original_request

*attribute* · *instance attribute*

```python
original_request: ExecutionRequest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L104-L104)

### particula.execution.fallback.FallbackResolution.policy

*attribute* · *instance attribute*

```python
policy: FallbackPolicy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L109-L109)

### particula.execution.fallback.FallbackResolution.requested_backend

*attribute* · *instance attribute*

```python
requested_backend: Backend
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L111-L111)

### particula.execution.fallback.FallbackResolution.selected_backend

*attribute* · *instance attribute*

```python
selected_backend: Backend
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L112-L112)

### particula.execution.fallback.FallbackResolution.__init__

*method*

```python
def __init__(original_request: ExecutionRequest, cpu_request: ExecutionRequest, original_error: ExecutionCapabilityError, cpu_state: CPUExecutionState, adapter: ExecutionAdapter, policy: FallbackPolicy, boundary: FallbackBoundary, requested_backend: Backend, selected_backend: Backend, capability_reason: ExecutionCapabilityReason) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L0-L0)

### particula.execution.fallback.FallbackResolution.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate resolution metadata without reselecting an adapter.

**Raises**

- `TypeError` — If a retained carrier or metadata field has an invalid
type.
- `ValueError` — If CPU request or error provenance is inconsistent.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L115-L141)

## particula.execution.fallback._disallow

*function*

```python
def _disallow(fallback: FallbackRequest, *, state: str, boundary: str) -> FallbackDisallowedError
```

Create a deterministic disallowed error for an invalid fallback path.

**Parameters**

- `fallback` (`FallbackRequest`) — Request whose original execution metadata is reported.
- `state` (`str`) — Deterministic state label describing the rejection.
- `boundary` (`str`) — Deterministic boundary label describing the rejection.

**Returns**

- (`FallbackDisallowedError`) — Error for the caller to chain from the original capability error.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L286-L310)

## particula.execution.fallback._fallback_metadata

*function*

```python
def _fallback_metadata(resolution: FallbackResolution) -> tuple[tuple[str, str], ...]
```

Return ordered immutable fallback provenance without copying results.

**Parameters**

- `resolution` (`FallbackResolution`) — Validated resolution supplying fallback provenance.

**Returns**

- (`tuple[str, str]`) — Ordered requested-backend, selected-backend, and capability-reason
- (`...`) — metadata pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L397-L413)

## particula.execution.fallback._is_static_execution_adapter

*function*

```python
def _is_static_execution_adapter(adapter: object) -> bool
```

Return whether an adapter statically exposes a callable execute seam.

**Parameters**

- `adapter` (`object`) — Object proposed as an execution adapter.

**Returns**

- (`bool`) — ``True`` when the object's static ``execute`` attribute is callable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L190-L202)

## particula.execution.fallback._validate_error_context

*function*

```python
def _validate_error_context(fallback: FallbackRequest) -> None
```

Require any supplied error context to describe the original request.

**Parameters**

- `fallback` (`FallbackRequest`) — Request and original error whose contexts must agree.

**Raises**

- `FallbackDisallowedError` — If supplied error context is inconsistent or
asserts a state or boundary this selector cannot verify.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L363-L394)

## particula.execution.fallback._validate_error_fields

*function*

```python
def _validate_error_fields(error: ExecutionCapabilityError) -> None
```

Validate the original capability error's closed carrier fields.

**Parameters**

- `error` (`ExecutionCapabilityError`) — Original typed capability error to validate.

**Raises**

- `TypeError` — If a reason or optional context field has an invalid type.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L337-L360)

## particula.execution.fallback._validate_fallback_enums

*function*

```python
def _validate_fallback_enums(policy: object, boundary: object, state_authority: object) -> None
```

Validate the closed fallback enum fields.

**Parameters**

- `policy` (`object`) — Proposed fallback policy.
- `boundary` (`object`) — Proposed caller-visible boundary.
- `state_authority` (`object`) — Proposed CPU-state authority assertion.

**Raises**

- `TypeError` — If any value is outside its corresponding fallback enum.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L263-L283)

## particula.execution.fallback._validate_request_and_error

*function*

```python
def _validate_request_and_error(fallback: FallbackRequest) -> None
```

Revalidate forged request and error carriers before context selection.

**Parameters**

- `fallback` (`FallbackRequest`) — Exact fallback carrier to revalidate.

**Raises**

- `TypeError` — If a carrier or its closed fields have invalid types.
- `ValueError` — If request backend and device backend differ.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L313-L334)

## particula.execution.fallback._validate_resolution_reason

*function*

```python
def _validate_resolution_reason(resolution: FallbackResolution) -> None
```

Validate that resolution provenance retains its original error reason.

**Parameters**

- `resolution` (`FallbackResolution`) — Resolution whose capability reason is checked.

**Raises**

- `TypeError` — If the capability reason has an invalid type.
- `ValueError` — If the reason differs from the original error reason.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L243-L260)

## particula.execution.fallback._validate_resolution_request

*function*

```python
def _validate_resolution_request(resolution: FallbackResolution) -> None
```

Validate that a resolution retains canonical CPU request metadata.

**Parameters**

- `resolution` (`FallbackResolution`) — Resolution whose request and backend fields are checked.

**Raises**

- `TypeError` — If a backend field has an invalid type.
- `ValueError` — If CPU request metadata is not canonical or preserved.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L205-L240)

## particula.execution.fallback.dispatch_cpu_fallback

*function*

```python
def dispatch_cpu_fallback(fallback: FallbackRequest) -> FallbackDispatchResult
```

Resolve and execute one CPU adapter without recovery or retry.

The adapter's native result and metadata remain unchanged. Provenance is
available only on the returned fallback dispatch carrier.

**Parameters**

- `fallback` (`FallbackRequest`) — Exact explicit fallback carrier to resolve and dispatch.

**Returns**

- (`FallbackDispatchResult`) — Immutable carrier retaining the resolution, native result, and fallback
- (`FallbackDispatchResult`) — provenance by identity.

**Raises**

- `TypeError` — If fallback or the adapter's result violates a carrier type.
- `ValueError` — If selection, result validation, or metadata validation
fails.
- `LookupError` — If no canonical CPU adapter is registered.
- `ExecutionCapabilityError` — The identical original error for default-deny
policy.
- `FallbackDisallowedError` — If fallback fails closed before selection.

Adapter exceptions propagate unchanged after the single adapter invocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L497-L526)

## particula.execution.fallback.resolve_cpu_fallback

*function*

```python
def resolve_cpu_fallback(fallback: FallbackRequest) -> FallbackResolution
```

Select the registered CPU adapter once for an eligible explicit request.

**Parameters**

- `fallback` (`FallbackRequest`) — Exact explicit fallback carrier.

**Returns**

- (`FallbackResolution`) — Immutable resolution retaining the selected adapter by identity.

**Raises**

- `TypeError` — If the carrier or its validated fields have invalid types.
- `ValueError` — If CPU capability lookup rejects the canonical request.
- `LookupError` — If no CPU adapter is registered for the canonical request.
- `ExecutionCapabilityError` — The identical original error for default-deny
policy.
- `FallbackDisallowedError` — If fallback fails closed before selection.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/fallback.py#L416-L494)
