"""Test the integral scale functions in the integral_scale_module.py file."""

import numpy as np
import pytest

from particula.gas.properties.integral_scale_module import (
    get_eulerian_integral_length,
    get_lagrangian_integral_time,
)


def test_get_lagrangian_integral_scale_scalar():
    """Test get_lagrangian_integral_scale with scalar inputs."""
    rms_velocity = 0.5  # m/s
    turbulent_dissipation = 1e-3  # m²/s³

    expected = (rms_velocity**2) / turbulent_dissipation
    result = get_lagrangian_integral_time(rms_velocity, turbulent_dissipation)

    assert np.isclose(result, expected, atol=1e-10)


def test_get_lagrangian_integral_scale_array():
    """Test get_lagrangian_integral_scale with NumPy array inputs."""
    rms_velocity = np.array([0.5, 0.8])
    turbulent_dissipation = np.array([1e-3, 2e-3])

    expected = (rms_velocity**2) / turbulent_dissipation
    result = get_lagrangian_integral_time(rms_velocity, turbulent_dissipation)

    assert np.allclose(result, expected, atol=1e-10)


def test_get_lagrangian_integral_scale_invalid_values():
    """Test get_lagrangian_integral_scale raises errors for invalid inputs."""
    with pytest.raises(ValueError):
        get_lagrangian_integral_time(-0.5, 1e-3)

    with pytest.raises(ValueError):
        get_lagrangian_integral_time(0.5, -1e-3)


def test_get_eulerian_integral_scale_scalar():
    """Test get_eulerian_integral_scale with scalar inputs."""
    rms_velocity = 0.5  # m/s
    turbulent_dissipation = 1e-3  # m²/s³

    expected = 0.5 * (rms_velocity**3) / turbulent_dissipation
    result = get_eulerian_integral_length(rms_velocity, turbulent_dissipation)

    assert np.isclose(result, expected, atol=1e-10)


def test_get_eulerian_integral_scale_array():
    """Test get_eulerian_integral_scale with NumPy array inputs."""
    rms_velocity = np.array([0.5, 0.8])
    turbulent_dissipation = np.array([1e-3, 2e-3])

    expected = 0.5 * (rms_velocity**3) / turbulent_dissipation
    result = get_eulerian_integral_length(rms_velocity, turbulent_dissipation)

    assert np.allclose(result, expected, atol=1e-10)


def test_get_eulerian_integral_scale_invalid_values():
    """Test get_eulerian_integral_scale raises errors for invalid inputs."""
    with pytest.raises(ValueError):
        get_eulerian_integral_length(-0.5, 1e-3)

    with pytest.raises(ValueError):
        get_eulerian_integral_length(0.5, -1e-3)


def test_integral_scales_edge_case():
    """Test integral scale functions with small values near precision."""
    rms_velocity = 1e-10
    turbulent_dissipation = 1e-10

    expected_tl = (rms_velocity**2) / turbulent_dissipation
    expected_le = 0.5 * (rms_velocity**3) / turbulent_dissipation

    result_tl = get_lagrangian_integral_time(
        rms_velocity, turbulent_dissipation
    )
    result_le = get_eulerian_integral_length(
        rms_velocity, turbulent_dissipation
    )

    assert np.isclose(result_tl, expected_tl, atol=1e-10)
    assert np.isclose(result_le, expected_le, atol=1e-10)
