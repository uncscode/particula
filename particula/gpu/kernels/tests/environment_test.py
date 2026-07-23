"""Tests for private GPU environment normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

wp: Any = None
try:
    import warp as wp
except ImportError:
    pass

pytestmark = (
    [pytest.mark.warp, pytest.mark.skip(reason="Warp not installed")]
    if wp is None
    else pytest.mark.warp
)

if wp is not None:
    from particula.gas.environment_data import EnvironmentData  # noqa: E402
    from particula.gpu.conversion import to_warp_environment_data  # noqa: E402
    from particula.gpu.kernels.environment import (  # noqa: E402
        _broadcast_scalar_array,
        _ensure_environment_arrays,
        _is_warp_array_like,
        _validate_box_array,
        validate_environment_inputs,
    )
    from particula.gpu.tests.cuda_availability import (  # noqa: E402
        cuda_available,
        warp_devices,
    )


def _available_warp_devices() -> list[str]:
    """Return collection-safe Warp device params."""
    if wp is None:
        return ["cpu"]
    return warp_devices(wp)


def _make_direct_input(
    value: float | list[float], *, device: str
) -> float | Any:
    """Build scalar or Warp-array test inputs lazily."""
    if isinstance(value, list):
        assert wp is not None
        return wp.array(value, dtype=wp.float64, device=device)
    return value


@pytest.fixture(params=_available_warp_devices())
def device(request) -> str:
    """Provide available Warp devices for testing."""
    return request.param


def _make_environment(n_boxes: int, n_species: int) -> EnvironmentData:
    """Create deterministic environment data."""
    return EnvironmentData(
        temperature=np.linspace(298.15, 300.15, n_boxes, dtype=np.float64),
        pressure=np.linspace(101325.0, 100000.0, n_boxes, dtype=np.float64),
        saturation_ratio=np.ones((n_boxes, n_species), dtype=np.float64),
    )


def test_environment_helper_broadcasts_scalar_inputs(device: str) -> None:
    """Scalar direct inputs broadcast to canonical ``(n_boxes,)`` arrays."""
    temperature, pressure = _ensure_environment_arrays(
        temperature=298.15,
        pressure=101325.0,
        environment=None,
        n_boxes=3,
        device=wp.get_device(device),
        caller_name="test_caller",
    )

    assert temperature.shape == (3,)
    assert pressure.shape == (3,)
    np.testing.assert_allclose(temperature.numpy(), [298.15, 298.15, 298.15])
    np.testing.assert_allclose(pressure.numpy(), [101325.0, 101325.0, 101325.0])


def test_environment_helper_returns_valid_direct_arrays_unchanged(
    device: str,
) -> None:
    """Valid direct Warp arrays are reused without copies."""
    temperature = wp.array([298.15, 299.15], dtype=wp.float64, device=device)
    pressure = wp.array([101325.0, 101000.0], dtype=wp.float64, device=device)

    returned_temperature, returned_pressure = _ensure_environment_arrays(
        temperature=temperature,
        pressure=pressure,
        environment=None,
        n_boxes=2,
        device=temperature.device,
        caller_name="test_caller",
    )

    assert returned_temperature is temperature
    assert returned_pressure is pressure


def test_environment_preflight_validates_arrays_without_scalar_broadcast(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Preflight accepts valid arrays without allocating scalar broadcasts."""
    temperature = wp.array([298.15, 299.15], dtype=wp.float64, device=device)
    pressure = wp.array([101325.0, 101000.0], dtype=wp.float64, device=device)

    def _unexpected_broadcast(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("preflight broadcast a scalar input")

    monkeypatch.setattr(
        "particula.gpu.kernels.environment._broadcast_scalar_array",
        _unexpected_broadcast,
    )
    validate_environment_inputs(
        temperature=temperature,
        pressure=pressure,
        environment=None,
        n_boxes=2,
        device=temperature.device,
        caller_name="test_caller",
    )


@pytest.mark.parametrize(
    ("name", "values", "raises"),
    [
        ("temperature", [298.15, 299.15], False),
        ("pressure", [101325.0, 0.0], True),
    ],
)
def test_environment_array_validation_avoids_input_materialization(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    name: str,
    values: list[float],
    raises: bool,
) -> None:
    """Validate device arrays with a scalar status readback, not ``.numpy()``."""
    temperature = wp.array([298.15, 299.15], dtype=wp.float64, device=device)
    pressure = wp.array([101325.0, 101000.0], dtype=wp.float64, device=device)
    supplied = wp.array(values, dtype=wp.float64, device=device)
    if name == "temperature":
        temperature = supplied
    else:
        pressure = supplied

    def _forbidden_numpy(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise AssertionError("input array was materialized on the host")

    monkeypatch.setattr(supplied, "numpy", _forbidden_numpy, raising=False)

    if raises:
        with pytest.raises(ValueError, match=f"{name} must be finite and > 0"):
            validate_environment_inputs(
                temperature=temperature,
                pressure=pressure,
                environment=None,
                n_boxes=2,
                device=wp.get_device(device),
                caller_name="test_caller",
            )
    else:
        validate_environment_inputs(
            temperature=temperature,
            pressure=pressure,
            environment=None,
            n_boxes=2,
            device=wp.get_device(device),
            caller_name="test_caller",
        )


def test_environment_helper_returns_environment_arrays_unchanged(
    device: str,
) -> None:
    """Valid ``WarpEnvironmentData`` arrays are reused without copies."""
    environment = to_warp_environment_data(
        _make_environment(2, 1), device=device
    )

    returned_temperature, returned_pressure = _ensure_environment_arrays(
        temperature=None,
        pressure=None,
        environment=environment,
        n_boxes=2,
        device=environment.temperature.device,
        caller_name="test_caller",
    )

    assert returned_temperature is environment.temperature
    assert returned_pressure is environment.pressure


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, [101325.0, 101000.0]),
        ([298.15, 299.15], 101325.0),
    ],
)
def test_environment_helper_accepts_hybrid_scalar_and_array_inputs(
    device: str,
    temperature: float | Any,
    pressure: float | Any,
) -> None:
    """Hybrid direct inputs broadcast only the scalar side."""
    temperature_value = _make_direct_input(temperature, device=device)
    pressure_value = _make_direct_input(pressure, device=device)

    returned_temperature, returned_pressure = _ensure_environment_arrays(
        temperature=temperature_value,
        pressure=pressure_value,
        environment=None,
        n_boxes=2,
        device=wp.get_device(device),
        caller_name="test_caller",
    )

    assert returned_temperature.shape == (2,)
    assert returned_pressure.shape == (2,)
    np.testing.assert_allclose(
        returned_temperature.numpy(),
        [298.15, 298.15]
        if not hasattr(temperature_value, "device")
        else [298.15, 299.15],
    )
    np.testing.assert_allclose(
        returned_pressure.numpy(),
        [101325.0, 101325.0]
        if not hasattr(pressure_value, "device")
        else [101325.0, 101000.0],
    )


def test_environment_helper_rejects_wrong_shape(device: str) -> None:
    """Direct arrays must match ``(n_boxes,)`` exactly."""
    temperature = wp.zeros((3,), dtype=wp.float64, device=device)

    with pytest.raises(ValueError, match=r"\(n_boxes,\)"):
        _validate_box_array(
            "temperature",
            temperature,
            n_boxes=2,
            device=temperature.device,
            caller_name="test_caller",
        )


def test_environment_helper_detects_warp_array_like_and_plain_inputs(
    device: str,
) -> None:
    """Warp-like arrays are detected and plain scalars are rejected."""
    temperature = wp.zeros((2,), dtype=wp.float64, device=device)

    assert _is_warp_array_like(temperature)
    assert not _is_warp_array_like(298.15)

    with pytest.raises(ValueError, match="must be a Warp array"):
        _validate_box_array(
            "temperature",
            298.15,
            n_boxes=2,
            device=temperature.device,
            caller_name="test_caller",
        )


class _FakeTensorLike:
    """Tensor-like object that should not satisfy the Warp-array contract."""

    def __init__(self, values: np.ndarray) -> None:
        self.shape = values.shape
        self.device = "cpu"
        self.dtype = values.dtype
        self._values = values

    def numpy(self) -> np.ndarray:
        """Return the stored array for compatibility-style probing."""
        return self._values


def test_environment_helper_rejects_non_warp_tensor_like_input() -> None:
    """Unsupported tensor-like inputs fail before any later Warp execution."""
    values = _FakeTensorLike(np.array([298.15, 299.15], dtype=np.float64))

    assert not _is_warp_array_like(values)

    with pytest.raises(ValueError, match="must be a Warp array"):
        _validate_box_array(
            "temperature",
            values,
            n_boxes=2,
            device=wp.get_device("cpu"),
            caller_name="test_caller",
        )


def test_environment_helper_rejects_integer_scalars(device: str) -> None:
    """Integer direct inputs are rejected before canonical float broadcast."""
    with pytest.raises(ValueError, match="floating scalar"):
        _ensure_environment_arrays(
            temperature=298,
            pressure=101325.0,
            environment=None,
            n_boxes=1,
            device=wp.get_device(device),
            caller_name="test_caller",
        )


def test_environment_helper_rejects_integer_dtype_arrays(device: str) -> None:
    """Only supported Warp float dtypes are accepted for direct arrays."""
    temperature = wp.array([298, 299], dtype=wp.int32, device=device)
    pressure = wp.array([101325.0, 101000.0], dtype=wp.float64, device=device)

    with pytest.raises(ValueError, match="supported Warp float dtype"):
        _ensure_environment_arrays(
            temperature=temperature,
            pressure=pressure,
            environment=None,
            n_boxes=2,
            device=wp.get_device(device),
            caller_name="test_caller",
        )


def test_environment_helper_reuses_scalar_broadcast_arrays(device: str) -> None:
    """Repeated scalar normalization reuses cached Warp broadcast arrays."""
    temperature_a, pressure_a = _ensure_environment_arrays(
        temperature=298.15,
        pressure=101325.0,
        environment=None,
        n_boxes=2,
        device=wp.get_device(device),
        caller_name="test_caller",
    )
    temperature_b, pressure_b = _ensure_environment_arrays(
        temperature=298.15,
        pressure=101325.0,
        environment=None,
        n_boxes=2,
        device=wp.get_device(device),
        caller_name="test_caller",
    )

    assert temperature_a is temperature_b
    assert pressure_a is pressure_b


def test_environment_helper_cached_scalar_broadcasts_do_not_leak_stale_values(
    device: str,
) -> None:
    """Changing a scalar input produces a distinct buffer with fresh values."""
    first = _broadcast_scalar_array(298.15, 2, wp.get_device(device))
    second = _broadcast_scalar_array(299.15, 2, wp.get_device(device))

    assert first is not second
    np.testing.assert_allclose(first.numpy(), [298.15, 298.15])
    np.testing.assert_allclose(second.numpy(), [299.15, 299.15])


@pytest.mark.cuda
def test_environment_helper_skips_host_readback_for_cuda_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA direct-array validation avoids implicit ``.numpy()`` readback."""
    if not cuda_available(wp):
        pytest.skip("CUDA not available for readback guard test")

    temperature = wp.array([298.15, 299.15], dtype=wp.float64, device="cuda")
    pressure = wp.array([101325.0, 101000.0], dtype=wp.float64, device="cuda")

    def _forbidden_numpy(self: Any) -> np.ndarray:
        raise AssertionError("unexpected host readback")

    monkeypatch.setattr(temperature, "numpy", _forbidden_numpy, raising=False)
    monkeypatch.setattr(pressure, "numpy", _forbidden_numpy, raising=False)

    returned_temperature, returned_pressure = _ensure_environment_arrays(
        temperature=temperature,
        pressure=pressure,
        environment=None,
        n_boxes=2,
        device=wp.get_device("cuda"),
        caller_name="test_caller",
    )

    assert returned_temperature is temperature
    assert returned_pressure is pressure


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (0.0, 101325.0, "temperature must be finite and > 0"),
        (-1.0, 101325.0, "temperature must be finite and > 0"),
        (float("nan"), 101325.0, "temperature must be finite and > 0"),
        (298.15, 0.0, "pressure must be finite and > 0"),
        (298.15, float("inf"), "pressure must be finite and > 0"),
    ],
)
def test_environment_helper_rejects_invalid_scalar_domains(
    device: str,
    temperature: float,
    pressure: float,
    message: str,
) -> None:
    """Scalar inputs must be positive finite physical values."""
    with pytest.raises(ValueError, match=message):
        _ensure_environment_arrays(
            temperature=temperature,
            pressure=pressure,
            environment=None,
            n_boxes=1,
            device=wp.get_device(device),
            caller_name="test_caller",
        )


@pytest.mark.parametrize(
    ("name", "values", "message"),
    [
        (
            "temperature",
            np.array([298.15, 0.0], dtype=np.float64),
            "temperature must be finite and > 0",
        ),
        (
            "temperature",
            np.array([298.15, np.nan], dtype=np.float64),
            "temperature must be finite and > 0",
        ),
        (
            "pressure",
            np.array([101325.0, np.inf], dtype=np.float64),
            "pressure must be finite and > 0",
        ),
    ],
)
def test_environment_helper_rejects_invalid_direct_array_domains(
    device: str,
    name: str,
    values: np.ndarray,
    message: str,
) -> None:
    """Direct Warp-array inputs must be positive finite physical values."""
    temperature = wp.array([298.15, 299.15], dtype=wp.float64, device=device)
    pressure = wp.array([101325.0, 101000.0], dtype=wp.float64, device=device)
    invalid = wp.array(values, dtype=wp.float64, device=device)
    if name == "temperature":
        temperature = invalid
    else:
        pressure = invalid

    with pytest.raises(ValueError, match=message):
        _ensure_environment_arrays(
            temperature=temperature,
            pressure=pressure,
            environment=None,
            n_boxes=2,
            device=wp.get_device(device),
            caller_name="test_caller",
        )


def test_environment_helper_rejects_invalid_environment_array_domains(
    device: str,
) -> None:
    """Environment-backed arrays must be positive finite physical values."""
    environment = to_warp_environment_data(
        _make_environment(2, 1), device=device
    )
    environment.pressure = wp.array(
        [101325.0, 0.0], dtype=wp.float64, device=device
    )

    with pytest.raises(
        ValueError,
        match="environment.pressure must be finite and > 0",
    ):
        _ensure_environment_arrays(
            temperature=None,
            pressure=None,
            environment=environment,
            n_boxes=2,
            device=environment.temperature.device,
            caller_name="test_caller",
        )


def test_environment_helper_returns_direct_array_dtype_unchanged(
    device: str,
) -> None:
    """Valid direct Warp arrays are returned without copying or coercion."""
    temperature = wp.array([298.15, 299.15], dtype=wp.float32, device=device)
    pressure = wp.array([101325.0, 101000.0], dtype=wp.float32, device=device)

    returned_temperature, returned_pressure = _ensure_environment_arrays(
        temperature=temperature,
        pressure=pressure,
        environment=None,
        n_boxes=2,
        device=temperature.device,
        caller_name="test_caller",
    )

    assert returned_temperature is temperature
    assert returned_pressure is pressure
    assert returned_temperature.dtype == wp.float32
    assert returned_pressure.dtype == wp.float32


def test_environment_helper_returns_environment_array_dtype_unchanged(
    device: str,
) -> None:
    """Environment-backed Warp arrays are returned without copying."""

    class _EnvironmentLike:
        def __init__(self) -> None:
            self.temperature = wp.array(
                [298.15, 299.15], dtype=wp.float32, device=device
            )
            self.pressure = wp.array(
                [101325.0, 101000.0], dtype=wp.float32, device=device
            )

    environment = _EnvironmentLike()

    returned_temperature, returned_pressure = _ensure_environment_arrays(
        temperature=None,
        pressure=None,
        environment=environment,
        n_boxes=2,
        device=environment.temperature.device,
        caller_name="test_caller",
    )

    assert returned_temperature is environment.temperature
    assert returned_pressure is environment.pressure
    assert returned_temperature.dtype == wp.float32
    assert returned_pressure.dtype == wp.float32


def test_environment_helper_rejects_environment_wrong_n_boxes(
    device: str,
) -> None:
    """Environment-backed arrays must match the requested ``n_boxes``."""
    environment = to_warp_environment_data(
        _make_environment(2, 1), device=device
    )

    with pytest.raises(ValueError, match=r"\(n_boxes,\)"):
        _ensure_environment_arrays(
            temperature=None,
            pressure=None,
            environment=environment,
            n_boxes=1,
            device=environment.temperature.device,
            caller_name="test_caller",
        )


@pytest.mark.cuda
def test_environment_helper_rejects_wrong_device(device: str) -> None:
    """Direct arrays on the wrong Warp device raise a stable error."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    temperature = wp.zeros((2,), dtype=wp.float64, device=device)

    with pytest.raises(
        ValueError, match="device does not match expected device"
    ):
        _validate_box_array(
            "temperature",
            temperature,
            n_boxes=2,
            device=wp.get_device(wrong_device),
            caller_name="test_caller",
        )


def test_environment_helper_rejects_mixed_direct_and_environment_inputs(
    device: str,
) -> None:
    """Direct inputs cannot be mixed with ``environment``."""
    environment = to_warp_environment_data(
        _make_environment(1, 1), device=device
    )

    with pytest.raises(
        ValueError,
        match="direct temperature/pressure inputs with environment",
    ):
        _ensure_environment_arrays(
            temperature=298.15,
            pressure=None,
            environment=environment,
            n_boxes=1,
            device=environment.temperature.device,
            caller_name="test_caller",
        )


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [(298.15, None), (None, 101325.0), (None, None)],
)
def test_environment_helper_rejects_missing_direct_inputs(
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Direct-input mode requires both temperature and pressure."""
    with pytest.raises(
        ValueError,
        match="temperature and pressure must both be provided",
    ):
        _ensure_environment_arrays(
            temperature=temperature,
            pressure=pressure,
            environment=None,
            n_boxes=1,
            device=wp.get_device(device),
            caller_name="test_caller",
        )
