"""Velocity correlation f2 test module."""

import numpy as np
import pytest

from particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_f2_ao2008 import (  # noqa: E501
    get_f2_longitudinal_velocity_correlation,
)


def test_compute_f2_longitudinal_velocity_correlation_scalar():
    """Test compute_f2_longitudinal_velocity_correlation with scalar inputs."""
    collisional_radius = 0.1  # [m]
    taylor_microscale = 0.05  # [m]
    eulerian_integral_length = 1.0  # [m]

    # Compute expected values using explicit computation
    beta = (np.sqrt(2) * taylor_microscale) / eulerian_integral_length
    sqrt_term_t = np.sqrt(1 - 2 * beta**2)
    denominator = 2 * sqrt_term_t

    exp_term_1 = np.exp(
        -2 * collisional_radius / ((1 + sqrt_term_t) * eulerian_integral_length)
    )
    exp_term_2 = np.exp(
        -2 * collisional_radius / ((1 - sqrt_term_t) * eulerian_integral_length)
    )

    expected = (1 / denominator) * (
        (1 + sqrt_term_t) * exp_term_1 - (1 - sqrt_term_t) * exp_term_2
    )

    result = get_f2_longitudinal_velocity_correlation(
        collisional_radius, taylor_microscale, eulerian_integral_length
    )

    assert np.isclose(result, expected, atol=1e-10), (
        f"Expected {expected}, but got {result}"
    )


def test_compute_f2_longitudinal_velocity_correlation_array():
    """Test compute_f2_longitudinal_velocity_correlation with arrays."""
    collisional_radius = np.array([0.1, 0.5, 1.0])  # [m]
    taylor_microscale = 0.05  # [m]
    eulerian_integral_length = 1.0  # [m]

    beta = (np.sqrt(2) * taylor_microscale) / eulerian_integral_length
    sqrt_term = np.sqrt(1 - 2 * beta**2)
    denominator = 2 * sqrt_term

    exp_term_1a = np.exp(
        -2 * collisional_radius / ((1 + sqrt_term) * eulerian_integral_length)
    )
    exp_term_2b = np.exp(
        -2 * collisional_radius / ((1 - sqrt_term) * eulerian_integral_length)
    )

    expected = (1 / denominator) * (
        (1 + sqrt_term) * exp_term_1a - (1 - sqrt_term) * exp_term_2b
    )

    result = get_f2_longitudinal_velocity_correlation(
        collisional_radius, taylor_microscale, eulerian_integral_length
    )

    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-10)


def test_invalid_inputs():
    """Ensure validation errors are raised for invalid inputs."""
    with pytest.raises(ValueError):
        get_f2_longitudinal_velocity_correlation(
            -0.1, 0.05, 1.0
        )  # Negative collisional_radius

    with pytest.raises(ValueError):
        get_f2_longitudinal_velocity_correlation(
            0.1, -0.05, 1.0
        )  # Negative taylor_microscale

    with pytest.raises(ValueError):
        get_f2_longitudinal_velocity_correlation(
            0.1, 0.05, -1.0
        )  # Negative eulerian_integral_length


def test_edge_cases():
    """Test compute_f2_longitudinal_velocity_correlation with extremes."""
    collisional_radius = np.array(
        [1e-6, 1e-3, 10.0]
    )  # Very small and large values
    taylor_microscale = 0.05  # [m]
    eulerian_integral_length = 1.0  # [m]

    result = get_f2_longitudinal_velocity_correlation(
        collisional_radius, taylor_microscale, eulerian_integral_length
    )

    assert np.all(np.isfinite(result)), "Expected all values to be finite"
