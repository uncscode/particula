"""Private GPU environment normalization helpers.

This module validates direct temperature and pressure inputs for GPU kernel
entry points and normalizes them into canonical Warp arrays with shape
``(n_boxes,)`` on the caller's device. Supported entry-point contracts accept
scalars, device-local Warp arrays, or a ``WarpEnvironmentData`` container
without performing hidden device transfers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - handled via import guards
    raise ImportError(
        "Warp is required for GPU environment helpers. "
        "Install with: pip install warp-lang"
    ) from exc


_SUPPORTED_WARP_FLOAT_DTYPES = (wp.float32, wp.float64)
_SCALAR_BROADCAST_CACHE: dict[tuple[str, int, float], Any] = {}
_SCALAR_BROADCAST_CACHE_MAXSIZE = 32


def _device_key(device: Any) -> str:
    """Return a stable cache key for a Warp device."""
    return str(device)


def _is_cpu_device(device: Any) -> bool:
    """Return ``True`` when a Warp device is CPU-backed."""
    return bool(
        getattr(device, "is_cpu", False) or str(device).startswith("cpu")
    )


def _is_supported_warp_float_dtype(dtype: Any) -> bool:
    """Return ``True`` for supported one-dimensional Warp float dtypes."""
    return any(dtype == supported for supported in _SUPPORTED_WARP_FLOAT_DTYPES)


def _is_warp_array_like(value: Any) -> bool:
    """Return ``True`` when ``value`` behaves like a Warp array.

    Args:
        value: Object to inspect.

    Returns:
        ``True`` when the object is a Warp array instance that exposes the
        expected runtime attributes.
    """
    warp_array_type = getattr(wp, "array", None)
    if isinstance(warp_array_type, type):
        try:
            if not isinstance(value, warp_array_type):
                return False
        except TypeError:
            return False

    value_type = type(value)
    return (
        value_type.__module__.startswith("warp")
        and value_type.__name__ == "array"
        and hasattr(value, "shape")
        and hasattr(value, "device")
        and hasattr(value, "dtype")
        and hasattr(value, "ptr")
        and hasattr(value, "ndim")
        and hasattr(value, "numpy")
        and callable(value.numpy)
    )


def _validate_positive_finite_scalar(
    name: str,
    value: float,
    caller_name: str,
) -> None:
    """Validate a scalar physical input domain before array creation."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0 in {caller_name}.")


def _validate_positive_finite_array(
    name: str,
    values: Any,
    caller_name: str,
) -> None:
    """Validate a per-box physical input domain before kernel launch.

    Notes:
        CPU-resident Warp arrays are checked elementwise. Device-resident Warp
        arrays are validated with reduction helpers, avoiding full-array host
        conversion via ``.numpy()``.
    """
    if not _is_cpu_device(values.device):
        values_np = np.asarray(type(values).numpy(values), dtype=np.float64)
        if not np.all(np.isfinite(values_np)) or np.any(values_np <= 0.0):
            raise ValueError(f"{name} must be finite and > 0 in {caller_name}.")
        return

    values_np = np.asarray(values.numpy(), dtype=np.float64)
    if not np.all(np.isfinite(values_np)) or np.any(values_np <= 0.0):
        raise ValueError(f"{name} must be finite and > 0 in {caller_name}.")


def _coerce_direct_float_scalar(
    name: str,
    value: Any,
    caller_name: str,
) -> float:
    """Coerce a supported direct scalar input into canonical ``float``."""
    if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
        if hasattr(value, "shape"):
            raise ValueError(
                f"{name} must be a scalar or Warp array with shape "
                f"(n_boxes,) in {caller_name}."
            )
        raise ValueError(
            f"{name} must be a floating scalar or Warp array with shape "
            f"(n_boxes,) in {caller_name}."
        )
    return float(value)


def _broadcast_scalar_array(value: float, n_boxes: int, device: Any) -> Any:
    """Broadcast a scalar into a cached canonical Warp ``float64`` array."""
    cache_key = (_device_key(device), n_boxes, value)
    cached = _SCALAR_BROADCAST_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if len(_SCALAR_BROADCAST_CACHE) >= _SCALAR_BROADCAST_CACHE_MAXSIZE:
        _SCALAR_BROADCAST_CACHE.clear()

    broadcast = wp.full(
        n_boxes,
        wp.float64(value),
        dtype=wp.float64,
        device=device,
    )
    _SCALAR_BROADCAST_CACHE[cache_key] = broadcast
    return broadcast


def _coerce_direct_scalar(
    name: str,
    value: Any,
    caller_name: str,
) -> float:
    """Coerce a supported direct scalar input into ``float``.

    Args:
        name: Human-readable scalar label for stable error messages.
        value: Candidate scalar value.
        caller_name: Entry-point name for stable contract messages.

    Returns:
        The coerced scalar value.

    Raises:
        ValueError: If ``value`` is not a supported scalar direct input.
    """
    try:
        return _coerce_direct_float_scalar(name, value, caller_name)
    except ValueError as exc:
        if hasattr(value, "shape"):
            raise ValueError(
                f"{name} must be a scalar or Warp array with shape "
                f"(n_boxes,) in {caller_name}."
            ) from exc
        raise ValueError(
            f"{name} must be a floating scalar or Warp array with shape "
            f"(n_boxes,) in {caller_name}."
        ) from exc


def _get_environment_array(
    environment: Any,
    field_name: str,
    caller_name: str,
) -> Any:
    """Fetch a required environment field with a stable contract error.

    Args:
        environment: Environment-like container passed by the caller.
        field_name: Required array attribute name.
        caller_name: Entry-point name for stable contract messages.

    Returns:
        The requested environment field.

    Raises:
        ValueError: If the required field is missing.
    """
    value = getattr(environment, field_name, None)
    if value is None:
        raise ValueError(
            f"environment.{field_name} must be a Warp array with shape "
            f"(n_boxes,) in {caller_name}."
        )
    return value


def _validate_box_array(
    name: str,
    values: Any,
    n_boxes: int,
    device: Any,
    caller_name: str,
) -> Any:
    """Validate a direct or environment-backed per-box Warp array.

    Args:
        name: Human-readable array label for error messages.
        values: Warp array-like object to validate.
        n_boxes: Expected number of boxes.
        device: Expected Warp device.
        caller_name: Entry-point name for stable contract messages.

    Returns:
        The validated Warp array.

    Raises:
        ValueError: If the input is not a Warp array on the expected device
            with shape ``(n_boxes,)``.

    Notes:
        Valid arrays preserve caller-owned buffers and are forwarded unchanged.
    """
    if not _is_warp_array_like(values):
        raise ValueError(
            f"{name} must be a Warp array with shape (n_boxes,) in "
            f"{caller_name}."
        )
    if values.shape != (n_boxes,):
        raise ValueError(
            f"{name} shape {values.shape} does not match expected (n_boxes,) "
            f"in {caller_name}."
        )
    value_device = getattr(values, "device", None)
    if value_device is None or str(value_device) != str(device):
        raise ValueError(
            f"{name} device does not match expected device in {caller_name}."
        )
    if len(values.shape) != 1 or not _is_supported_warp_float_dtype(
        values.dtype
    ):
        raise ValueError(
            f"{name} must use a supported Warp float dtype in {caller_name}."
        )
    return values


def _ensure_environment_arrays(
    temperature: float | Any | None,
    pressure: float | Any | None,
    environment: Any | None,
    n_boxes: int,
    device: Any,
    caller_name: str,
) -> tuple[Any, Any]:
    """Normalize environment inputs into validated ``(n_boxes,)`` Warp arrays.

    Args:
        temperature: Scalar temperature [K], Warp array, or None.
        pressure: Scalar pressure [Pa], Warp array, or None.
        environment: Optional ``WarpEnvironmentData`` container with
            ``temperature`` and ``pressure`` arrays.
        n_boxes: Expected number of boxes.
        device: Expected Warp device.
        caller_name: Entry-point name for stable contract messages.

    Returns:
        Canonical temperature and pressure Warp arrays on ``device``.

    Raises:
        ValueError: If direct inputs are mixed with ``environment``.
        ValueError: If ``environment`` is omitted and one direct input is
            missing.
        ValueError: If any array shape is not ``(n_boxes,)``.
        ValueError: If any array device mismatches ``device``.

    Notes:
        When one direct input is scalar and the other is already a valid Warp
        array, only the scalar side is broadcast. Valid Warp arrays are
        forwarded unchanged.
    """
    if environment is not None:
        if temperature is not None or pressure is not None:
            raise ValueError(
                "Cannot mix direct temperature/pressure inputs with "
                f"environment in {caller_name}."
            )
        environment_temperature = _get_environment_array(
            environment,
            "temperature",
            caller_name,
        )
        environment_pressure = _get_environment_array(
            environment,
            "pressure",
            caller_name,
        )
        temperature_array = _validate_box_array(
            "environment.temperature",
            environment_temperature,
            n_boxes,
            device,
            caller_name,
        )
        pressure_array = _validate_box_array(
            "environment.pressure",
            environment_pressure,
            n_boxes,
            device,
            caller_name,
        )
        _validate_positive_finite_array(
            "environment.temperature",
            temperature_array,
            caller_name,
        )
        _validate_positive_finite_array(
            "environment.pressure",
            pressure_array,
            caller_name,
        )
        return temperature_array, pressure_array

    if temperature is None or pressure is None:
        raise ValueError(
            "temperature and pressure must both be provided when environment "
            f"is omitted in {caller_name}."
        )

    if _is_warp_array_like(temperature):
        temperature_array = _validate_box_array(
            "temperature",
            temperature,
            n_boxes,
            device,
            caller_name,
        )
        _validate_positive_finite_array(
            "temperature", temperature_array, caller_name
        )
    else:
        temperature_scalar = _coerce_direct_scalar(
            "temperature", temperature, caller_name
        )
        _validate_positive_finite_scalar(
            "temperature", temperature_scalar, caller_name
        )
        temperature_array = _broadcast_scalar_array(
            temperature_scalar,
            n_boxes,
            device,
        )

    if _is_warp_array_like(pressure):
        pressure_array = _validate_box_array(
            "pressure",
            pressure,
            n_boxes,
            device,
            caller_name,
        )
        _validate_positive_finite_array("pressure", pressure_array, caller_name)
    else:
        pressure_scalar = _coerce_direct_scalar(
            "pressure", pressure, caller_name
        )
        _validate_positive_finite_scalar(
            "pressure", pressure_scalar, caller_name
        )
        pressure_array = _broadcast_scalar_array(
            pressure_scalar,
            n_boxes,
            device,
        )

    return temperature_array, pressure_array
