"""Tests for private GPU environment normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from particula.gas.environment_data import EnvironmentData  # noqa: E402
from particula.gpu.conversion import to_warp_environment_data  # noqa: E402
from particula.gpu.kernels.environment import (  # noqa: E402
    _ensure_environment_arrays,
    _is_warp_array_like,
    _validate_box_array,
)
from particula.gpu.tests.cuda_availability import (  # noqa: E402
    cuda_available,
    warp_devices,
)


@pytest.fixture(params=warp_devices(wp))
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
        (
            298.15,
            wp.array([101325.0, 101000.0], dtype=wp.float64, device="cpu"),
        ),
        (
            wp.array([298.15, 299.15], dtype=wp.float64, device="cpu"),
            101325.0,
        ),
    ],
)
def test_environment_helper_accepts_hybrid_scalar_and_array_inputs(
    device: str,
    temperature: float | Any,
    pressure: float | Any,
) -> None:
    """Hybrid direct inputs broadcast only the scalar side."""
    temperature_value: Any = temperature
    pressure_value: Any = pressure
    if device != "cpu":
        if hasattr(temperature_value, "device"):
            temperature_value = wp.array(
                temperature_value.numpy(),
                dtype=wp.float64,
                device=device,
            )
        if hasattr(pressure_value, "device"):
            pressure_value = wp.array(
                pressure_value.numpy(),
                dtype=wp.float64,
                device=device,
            )

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
