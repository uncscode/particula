"""Tests for the velocity_correlation_terms module."""

import numpy as np
import pytest

from particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008 import (  # noqa: E501
    compute_b1,
    compute_b2,
    compute_beta,
    compute_c1,
    compute_c2,
    compute_d1,
    compute_d2,
    compute_e1,
    compute_e2,
    compute_z,
)


def test_compute_z():
    """Test compute_z with scalar and array inputs."""
    lagrangian_integral_time = np.array([1.0, 2.0, 3.0])  # [s]
    eulerian_integral_length = np.array([10.0, 20.0, 30.0])  # [m]

    expected = lagrangian_integral_time / eulerian_integral_length
    result = compute_z(lagrangian_integral_time, eulerian_integral_length)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_beta():
    """Test compute_beta with scalar and array inputs."""
    taylor_microscale = np.array([1.0, 2.0, 3.0])  # [m]
    eulerian_integral_length = np.array([10.0, 20.0, 30.0])  # [m]

    expected = (np.sqrt(2) * taylor_microscale) / eulerian_integral_length
    result = compute_beta(taylor_microscale, eulerian_integral_length)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_b1():
    """Test compute_b1 with valid inputs."""
    z = np.array([0.1, 0.2, 0.3])  # [-]
    expected = (1 + np.sqrt(1 - 2 * z**2)) / (2 * np.sqrt(1 - 2 * z**2))
    result = compute_b1(z)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_b2():
    """Test compute_b2 with valid inputs."""
    z = np.array([0.1, 0.2, 0.3])  # [-]
    expected = (1 - np.sqrt(1 - 2 * z**2)) / (2 * np.sqrt(1 - 2 * z**2))
    result = compute_b2(z)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_c1():
    """Test compute_c1 with valid inputs."""
    z = np.array([0.1, 0.2, 0.3])  # [-]
    lagrangian_integral_time = np.array([10.0, 20.0, 30.0])  # [s]

    expected = ((1 + np.sqrt(1 - 2 * z**2)) * lagrangian_integral_time) / 2
    result = compute_c1(z, lagrangian_integral_time)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_c2():
    """Test compute_c2 with valid inputs."""
    z = np.array([0.1, 0.2, 0.3])  # [-]
    lagrangian_integral_time = np.array([10.0, 20.0, 30.0])  # [s]

    expected = ((1 - np.sqrt(1 - 2 * z**2)) * lagrangian_integral_time) / 2
    result = compute_c2(z, lagrangian_integral_time)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_d1():
    """Test compute_d1 with valid inputs."""
    beta = np.array([0.1, 0.2, 0.3])  # [-]

    expected = (1 + np.sqrt(1 - 2 * beta**2)) / (2 * np.sqrt(1 - 2 * beta**2))
    result = compute_d1(beta)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_d2():
    """Test compute_d2 with valid inputs."""
    beta = np.array([0.1, 0.2, 0.3])  # [-]

    expected = (1 - np.sqrt(1 - 2 * beta**2)) / (2 * np.sqrt(1 - 2 * beta**2))
    result = compute_d2(beta)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_e1():
    """Test compute_e1 with valid inputs."""
    beta = np.array([0.1, 0.2, 0.3])  # [-]
    eulerian_integral_length = np.array([10.0, 20.0, 30.0])  # [m]

    expected = ((1 + np.sqrt(1 - 2 * beta**2)) * eulerian_integral_length) / 2
    result = compute_e1(beta, eulerian_integral_length)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_e2():
    """Test compute_e2 with valid inputs."""
    beta = np.array([0.1, 0.2, 0.3])  # [-]
    eulerian_integral_length = np.array([10.0, 20.0, 30.0])  # [m]

    expected = ((1 - np.sqrt(1 - 2 * beta**2)) * eulerian_integral_length) / 2
    result = compute_e2(beta, eulerian_integral_length)

    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_invalid_inputs():
    """Ensure validation errors are raised for invalid inputs."""
    with pytest.raises(ValueError):
        compute_z(-1, 10)  # Negative lagrangian_integral_time

    with pytest.raises(ValueError):
        compute_beta(1, -10)  # Negative length scale

    with pytest.raises(ValueError):
        compute_b1(-0.5)  # Invalid z value

    with pytest.raises(ValueError):
        compute_d1(-0.5)  # Invalid beta value

    with pytest.raises(ValueError):
        compute_e1(0.5, -10)  # Negative length scale


def test_edge_cases():
    """Test edge cases such as very small and very large values."""
    lagrangian_integral_time = np.array(
        [1e-6, 1e-3, 10.0]
    )  # Very small and large inertia values
    eulerian_integral_length = np.array(
        [1e-6, 1e-3, 10.0]
    )  # Very small and large values

    z_values = compute_z(lagrangian_integral_time, eulerian_integral_length)
    beta_values = compute_beta(
        lagrangian_integral_time, eulerian_integral_length
    )

    assert np.all(np.isfinite(z_values)), "Expected all z values to be finite"
    assert np.all(np.isfinite(beta_values)), (
        "Expected all beta values to be finite"
    )
