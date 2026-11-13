# particula.gpu.kernels.environment

*module*

Private GPU environment normalization helpers.

This module validates direct temperature and pressure inputs for GPU kernel
entry points and normalizes them into canonical Warp arrays with shape
``(n_boxes,)`` on the caller's device. Supported entry-point contracts accept
scalars, device-local Warp arrays, or a ``WarpEnvironmentData`` container
without hidden device transfers. ``validate_environment_inputs`` is the
read-only preflight phase: it completes before scalar broadcasting and does
not mutate caller-owned environment arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L1-L1)

## particula.gpu.kernels.environment._SCALAR_BROADCAST_CACHE

*attribute* · *module attribute*

```python
_SCALAR_BROADCAST_CACHE: dict[tuple[str, int, float], Any] = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L30-L30)

## particula.gpu.kernels.environment._SCALAR_BROADCAST_CACHE_MAXSIZE

*attribute* · *module attribute*

```python
_SCALAR_BROADCAST_CACHE_MAXSIZE = 32
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L31-L31)

## particula.gpu.kernels.environment._SUPPORTED_WARP_FLOAT_DTYPES

*attribute* · *module attribute*

```python
_SUPPORTED_WARP_FLOAT_DTYPES = (wp.float32, wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L29-L29)

## particula.gpu.kernels.environment._broadcast_scalar_array

*function*

```python
def _broadcast_scalar_array(value: float, n_boxes: int, device: Any) -> Any
```

Broadcast a scalar into a cached canonical Warp ``float64`` array.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L162-L179)

## particula.gpu.kernels.environment._coerce_direct_float_scalar

*function*

```python
def _coerce_direct_float_scalar(name: str, value: Any, caller_name: str) -> float
```

Coerce a supported direct scalar input into canonical ``float``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L143-L159)

## particula.gpu.kernels.environment._coerce_direct_scalar

*function*

```python
def _coerce_direct_scalar(name: str, value: Any, caller_name: str) -> float
```

Coerce a supported direct scalar input into ``float``.

**Parameters**

- `name` (`str`) — Human-readable scalar label for stable error messages.
- `value` (`Any`) — Candidate scalar value.
- `caller_name` (`str`) — Entry-point name for stable contract messages.

**Returns**

- (`float`) — The coerced scalar value.

**Raises**

- `ValueError` — If ``value`` is not a supported scalar direct input.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L182-L211)

## particula.gpu.kernels.environment._device_key

*function*

```python
def _device_key(device: Any) -> str
```

Return a stable cache key for a Warp device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L56-L58)

## particula.gpu.kernels.environment._ensure_environment_arrays

*function*

```python
def _ensure_environment_arrays(temperature: float | Any | None, pressure: float | Any | None, environment: Any | None, n_boxes: int, device: Any, caller_name: str) -> tuple[Any, Any]
```

Normalize environment inputs into validated ``(n_boxes,)`` Warp arrays.

**Parameters**

- `temperature` (`float | Any | None`) — Scalar temperature [K], Warp array, or None.
- `pressure` (`float | Any | None`) — Scalar pressure [Pa], Warp array, or None.
- `environment` (`Any | None`) — Optional ``WarpEnvironmentData`` container with ``temperature`` and ``pressure`` arrays.
- `n_boxes` (`int`) — Expected number of boxes.
- `device` (`Any`) — Expected Warp device.
- `caller_name` (`str`) — Entry-point name for stable contract messages.

**Returns**

- (`tuple[Any, Any]`) — Canonical temperature and pressure Warp arrays on ``device``.

**Raises**

- `ValueError` — If direct inputs are mixed with ``environment``.
- `ValueError` — If ``environment`` is omitted and one direct input is
missing.
- `ValueError` — If any array shape is not ``(n_boxes,)``.
- `ValueError` — If any array device mismatches ``device``.

> **Notes**
>
> When one direct input is scalar and the other is already a valid Warp
> array, only the scalar side is broadcast. Valid Warp arrays are
> forwarded unchanged.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L291-L419)

## particula.gpu.kernels.environment._get_environment_array

*function*

```python
def _get_environment_array(environment: Any, field_name: str, caller_name: str) -> Any
```

Fetch a required environment field with a stable contract error.

**Parameters**

- `environment` (`Any`) — Environment-like container passed by the caller.
- `field_name` (`str`) — Required array attribute name.
- `caller_name` (`str`) — Entry-point name for stable contract messages.

**Returns**

- (`Any`) — The requested environment field.

**Raises**

- `ValueError` — If the required field is missing.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L214-L238)

## particula.gpu.kernels.environment._is_cpu_device

*function*

```python
def _is_cpu_device(device: Any) -> bool
```

Return ``True`` when a Warp device is CPU-backed.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L61-L65)

## particula.gpu.kernels.environment._is_supported_warp_float_dtype

*function*

```python
def _is_supported_warp_float_dtype(dtype: Any) -> bool
```

Return ``True`` for supported one-dimensional Warp float dtypes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L68-L70)

## particula.gpu.kernels.environment._is_warp_array_like

*function*

```python
def _is_warp_array_like(value: Any) -> bool
```

Return ``True`` when ``value`` behaves like a Warp array.

**Parameters**

- `value` (`Any`) — Object to inspect.

**Returns**

- (`bool`) — ``True`` when the object is a Warp array instance that exposes the
- (`bool`) — expected runtime attributes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L73-L102)

## particula.gpu.kernels.environment._scan_positive_finite_float32

*function*

```python
def _scan_positive_finite_float32(values: wp.array(dtype=wp.float32), invalid: wp.array(dtype=wp.int32)) -> None
```

Record invalid positive physical values in a float32 array.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L34-L42)

## particula.gpu.kernels.environment._scan_positive_finite_float64

*function*

```python
def _scan_positive_finite_float64(values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Record invalid positive physical values in a float64 array.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L45-L53)

## particula.gpu.kernels.environment._validate_box_array

*function*

```python
def _validate_box_array(name: str, values: Any, n_boxes: int, device: Any, caller_name: str) -> Any
```

Validate a direct or environment-backed per-box Warp array.

**Parameters**

- `name` (`str`) — Human-readable array label for error messages.
- `values` (`Any`) — Warp array-like object to validate.
- `n_boxes` (`int`) — Expected number of boxes.
- `device` (`Any`) — Expected Warp device.
- `caller_name` (`str`) — Entry-point name for stable contract messages.

**Returns**

- (`Any`) — The validated Warp array.

**Raises**

- `ValueError` — If the input is not a Warp array on the expected device
with shape ``(n_boxes,)``.

> **Notes**
>
> Valid arrays preserve caller-owned buffers and are forwarded unchanged.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L241-L288)

## particula.gpu.kernels.environment._validate_positive_finite_array

*function*

```python
def _validate_positive_finite_array(name: str, values: Any, caller_name: str) -> None
```

Validate a per-box physical input domain before kernel launch.

> **Notes**
>
> The check scans values on their device, then reads back one scalar
> status flag without modifying or replacing the supplied Warp array.
> It is used by the preflight and normalization phases.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L115-L140)

## particula.gpu.kernels.environment._validate_positive_finite_scalar

*function*

```python
def _validate_positive_finite_scalar(name: str, value: float, caller_name: str) -> None
```

Validate a scalar physical input domain before array creation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L105-L112)

## particula.gpu.kernels.environment.validate_environment_inputs

*function*

```python
def validate_environment_inputs(temperature: float | Any | None, pressure: float | Any | None, environment: Any | None, n_boxes: int, device: Any, caller_name: str) -> None
```

Validate an environment source without broadcasting scalar inputs.

**Parameters**

- `temperature` (`float | Any | None`) — Direct scalar or ``(n_boxes,)`` Warp temperature [K].
- `pressure` (`float | Any | None`) — Direct scalar or ``(n_boxes,)`` Warp pressure [Pa].
- `environment` (`Any | None`) — Optional container providing both Warp arrays.
- `n_boxes` (`int`) — Expected number of spatial boxes.
- `device` (`Any`) — Expected Warp device for direct or container arrays.
- `caller_name` (`str`) — Entry-point name used in validation errors.

**Raises**

- `ValueError` — If sources are mixed or missing, or if an input has an
invalid scalar domain, Warp-array schema, device, or shape.

> **Notes**
>
> This is the read-only preflight half of
> :func:`_ensure_environment_arrays`. It validates all input domains
> before scalar broadcasting and does not mutate or replace caller-owned
> arrays or environment fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/environment.py#L422-L482)
