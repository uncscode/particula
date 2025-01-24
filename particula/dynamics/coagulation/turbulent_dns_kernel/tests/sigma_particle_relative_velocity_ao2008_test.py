"""
Tests for the sigma_relative_velocity_variance module.
"""

import pytest
import numpy as np

from particula.dynamics.coagulation.turbulent_dns_kernel.sigma_particle_relative_velocity_ao2008 import (
    get_relative_velocity_variance,
)


def test_get_relative_velocity_variance_array():
    """
    Test get_relative_velocity_variance with NumPy array inputs.
    """
    turbulence_intensity = 0.05  # [m/s]
    collisional_radius = np.array([0.05, 0.1, 0.2])  # [m]
    particle_inertia_time = np.array([2, 3, 5]) * 1e-6  # [s]
    particle_velocity = np.array([0.3, 0.4, 0.5])  # [m/s]
    taylor_microscale = 0.05  # [m]
    eulerian_integral_length = 1.0  # [m]
    lagrangian_integral_time = 0.2  # [s]

    expected_shape = (3, 3)
    result = get_relative_velocity_variance(
        turbulence_intensity,
        collisional_radius,
        particle_inertia_time,
        particle_velocity,
        taylor_microscale,
        eulerian_integral_length,
        lagrangian_integral_time,
    )

    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {result.shape}"


def test_invalid_inputs():
    """
    Ensure validation errors are raised for invalid inputs.
    """
    with pytest.raises(ValueError):
        get_relative_velocity_variance(
            -0.5, 0.05, 0.02, 0.3, 0.05, 1.0, 0.2
        )  # Negative turbulence_intensity

    with pytest.raises(ValueError):
        get_relative_velocity_variance(
            0.5, -0.05, 0.02, 0.3, 0.05, 1.0, 0.2
        )  # Negative collisional_radius

    with pytest.raises(ValueError):
        get_relative_velocity_variance(
            0.5, 0.05, -0.02, 0.3, 0.05, 1.0, 0.2
        )  # Negative particle_inertia_time

    with pytest.raises(ValueError):
        get_relative_velocity_variance(
            0.5, 0.05, 0.02, -0.3, 0.05, 1.0, 0.2
        )  # Negative particle_velocity


def test_edge_cases():
    """
    Test compute_relative_velocity_variance with extreme values.
    """
    turbulence_intensity = np.array(
        [1e-6, 1e-3, 10.0]
    )  # Very small and large values
    collisional_radius = np.array(
        [1e-6, 1e-3, 10.0]
    )  # Very small and large values
    particle_inertia_time = np.array(
        [1e-6, 1e-3, 10.0]
    )  # Very small and large values
    particle_velocity = np.array(
        [1e-6, 1e-3, 10.0]
    )  # Very small and large values

    result = get_relative_velocity_variance(
        turbulence_intensity=turbulence_intensity,
        collisional_radius=collisional_radius,
        particle_inertia_time=particle_inertia_time,
        particle_velocity=particle_velocity,
        taylor_microscale=0.05,
        eulerian_integral_length=1.0,
        lagrangian_integral_time=0.2,
    )

    assert np.all(np.isfinite(result)), "Expected all values to be finite"
