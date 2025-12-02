"""Test psi function from Ayala et al. (2008)."""

import numpy as np
import pytest

from particula.dynamics.coagulation.turbulent_dns_kernel.psi_ao2008 import (
    get_psi_ao2008,
)


def test_get_psi_ao2008_scalar():
    """Test get_psi_ao2008 with scalar inputs."""
    alpha = 2.0  # Turbulence parameter [-]
    phi = 1.0  # Characteristic velocity [m/s]
    particle_inertia_time = 0.05  # [s]
    particle_velocity = 0.1  # [m/s]

    expected = 1 / (
        (1 / particle_inertia_time) + (1 / alpha) + (particle_velocity / phi)
    ) - (
        particle_velocity
        / (
            2
            * phi
            * (
                (1 / particle_inertia_time)
                + (1 / alpha)
                + (particle_velocity / phi)
            )
            ** 2
        )
    )

    result = get_psi_ao2008(
        alpha, phi, particle_inertia_time, particle_velocity
    )

    assert np.isclose(result, expected, atol=1e-10), (
        f"Expected {expected}, but got {result}"
    )


def test_get_psi_ao2008_array():
    """Test get_psi_ao2008 with NumPy array inputs."""
    alpha = 2.0  # Turbulence parameter [-]
    phi = 1.0  # Characteristic velocity [m/s]
    particle_inertia_time = np.array([0.05, 0.1, 0.2])  # [s]
    particle_velocity = np.array([0.1, 0.2, 0.3])  # [m/s]

    denominator = (
        (1 / particle_inertia_time) + (1 / alpha) + (particle_velocity / phi)
    )
    expected = 1 / denominator - (
        particle_velocity / (2 * phi * denominator**2)
    )

    result = get_psi_ao2008(
        alpha, phi, particle_inertia_time, particle_velocity
    )

    assert result.shape == expected.shape, (
        f"Expected shape {expected.shape}, but got {result.shape}"
    )
    assert np.allclose(result, expected, atol=1e-10), (
        f"Expected {expected}, but got {result}"
    )


def test_get_psi_ao2008_invalid_inputs():
    """Test that get_psi_ao2008 raises validation errors for invalid inputs."""
    alpha = 2.0  # [-]
    phi = 1.0  # [m/s]
    particle_inertia_time = np.array([0.05, 0.1, 0.2])  # [s]
    particle_velocity = np.array([0.1, 0.2, 0.3])  # [m/s]

    with pytest.raises(ValueError):
        get_psi_ao2008(
            -alpha, phi, particle_inertia_time, particle_velocity
        )  # Negative alpha

    with pytest.raises(ValueError):
        get_psi_ao2008(
            alpha, -phi, particle_inertia_time, particle_velocity
        )  # Negative phi

    with pytest.raises(ValueError):
        get_psi_ao2008(
            alpha, phi, -particle_inertia_time, particle_velocity
        )  # Negative inertia time

    with pytest.raises(ValueError):
        get_psi_ao2008(
            alpha, phi, particle_inertia_time, -particle_velocity
        )  # Negative velocity


def test_get_psi_ao2008_edge_cases():
    """Test get_psi_ao2008 with extreme values such as very small or very large
    values.
    """
    alpha = 2.0  # [-]
    phi = 1.0  # [m/s]
    particle_inertia_time = np.array(
        [1e-6, 1e-3, 10.0]
    )  # Very small and large inertia values
    particle_velocity = np.array(
        [1e-6, 1e-3, 10.0]
    )  # Very small and large velocities

    result = get_psi_ao2008(
        alpha, phi, particle_inertia_time, particle_velocity
    )

    assert np.all(result >= 0), "Expected all values to be non-negative"
    assert np.isfinite(result).all(), "Expected all values to be finite"
