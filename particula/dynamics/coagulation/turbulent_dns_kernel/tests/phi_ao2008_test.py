"""Tests for the phi_function module."""

import numpy as np
import pytest

from particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008 import (
    get_phi_ao2008,
)


def test_get_phi_ao2008_scalar():
    """Test get_phi_ao2008 with scalar inputs."""
    alpha = 2.0  # Turbulence parameter [-]
    phi = 1.0  # Characteristic velocity [m/s]
    particle_inertia_time = np.array([0.05, 0.02])  # [s]
    particle_velocity = np.array([0.3, 0.1])  # [m/s]

    expected_shape = (2, 2)
    result = get_phi_ao2008(
        alpha, phi, particle_inertia_time, particle_velocity
    )

    assert result.shape == expected_shape, (
        f"Expected shape {expected_shape}, but got {result.shape}"
    )
    assert np.all(result >= 0), "Expected all values to be non-negative"


def test_get_phi_ao2008_array():
    """Test get_phi_ao2008 with NumPy array inputs."""
    alpha = 2.0  # Turbulence parameter [-]
    phi = 1.0  # Characteristic velocity [m/s]
    particle_inertia_time = np.array([0.05, 0.1, 0.2])  # [s]
    particle_velocity = np.array([0.1, 0.2, 0.3])  # [m/s]

    expected_shape = (3, 3)
    result = get_phi_ao2008(
        alpha, phi, particle_inertia_time, particle_velocity
    )

    assert result.shape == expected_shape, (
        f"Expected shape {expected_shape}, but got {result.shape}"
    )
    assert np.all(result >= 0), "Expected all values to be non-negative"


def test_get_phi_ao2008_invalid_inputs():
    """Test that get_phi_ao2008 raises validation errors for invalid inputs."""
    alpha = 2.0  # [-]
    phi = 1.0  # [m/s]
    particle_inertia_time = np.array([0.05, 0.1, 0.5])  # [s]
    particle_velocity = np.array([0.1, 0.2, 0.6])  # [m/s]

    with pytest.raises(ValueError):
        get_phi_ao2008(
            -alpha, phi, particle_inertia_time, particle_velocity
        )  # Negative alpha

    with pytest.raises(ValueError):
        get_phi_ao2008(
            alpha, -phi, particle_inertia_time, particle_velocity
        )  # Negative phi

    with pytest.raises(ValueError):
        get_phi_ao2008(
            alpha, phi, -particle_inertia_time, particle_velocity
        )  # Negative inertia time

    with pytest.raises(ValueError):
        get_phi_ao2008(
            alpha, phi, particle_inertia_time, -particle_velocity
        )  # Negative velocity


def test_get_phi_ao2008_edge_cases():
    """Test get_phi_ao2008 with extreme values such as very small or very large
    values.
    """
    alpha = 2.0  # [-]
    phi = 1.0  # [m/s]
    particle_inertia_time = np.array(
        [5e-6, 1e-3, 20.0]
    )  # Very small and large inertia values
    particle_velocity = np.array(
        [2e-6, 1e-3, 20.0]
    )  # Very small and large velocities

    result = get_phi_ao2008(
        alpha, phi, particle_inertia_time, particle_velocity
    )

    assert np.all(result >= 0), "Expected all values to be non-negative"
    assert np.isfinite(result).all(), "Expected all values to be finite"
