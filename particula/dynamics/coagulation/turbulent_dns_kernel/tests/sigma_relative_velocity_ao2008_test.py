"""Tests for the sigma_relative_velocity_variance module."""

import numpy as np
import pytest

from particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008 import (  # noqa: E501
    get_relative_velocity_variance,
)


def test_get_relative_velocity_variance_array():
    """Test get_relative_velocity_variance with NumPy array inputs."""
    expected_shape = (3, 3)
    result = get_relative_velocity_variance(
        fluid_rms_velocity=0.05,
        collisional_radius=np.array([0.05, 0.1, 0.2]),
        particle_inertia_time=np.array([2, 3, 5]) * 1e-6,
        particle_velocity=np.array([0.3, 0.4, 0.5]),
        taylor_microscale=0.05,
        eulerian_integral_length=2.0,
        lagrangian_integral_time=0.3,
        lagrangian_taylor_microscale_time=0.1,
    )

    assert result.shape == expected_shape


def test_invalid_inputs():
    """Ensure validation errors are raised for invalid inputs."""
    with pytest.raises(ValueError):
        get_relative_velocity_variance(
            -0.5,
            0.05,
            0.02,
            0.3,
            0.05,
            1.0,
            0.2,
            lagrangian_taylor_microscale_time=0.1,
        )  # Negative fluid_rms_velocity

    with pytest.raises(ValueError):
        get_relative_velocity_variance(
            0.5,
            -0.05,
            0.02,
            0.3,
            0.05,
            1.0,
            0.2,
            lagrangian_taylor_microscale_time=0.1,
        )  # Negative collisional_radius

    with pytest.raises(ValueError):
        get_relative_velocity_variance(
            0.5,
            0.05,
            -0.02,
            0.3,
            0.05,
            1.0,
            0.2,
            lagrangian_taylor_microscale_time=0.1,
        )  # Negative particle_inertia_time

    with pytest.raises(ValueError):
        get_relative_velocity_variance(
            0.5,
            0.05,
            0.02,
            -0.3,
            0.05,
            1.0,
            0.2,
            lagrangian_taylor_microscale_time=0.1,
        )  # Negative particle_velocity


def test_edge_cases():
    """Test compute_relative_velocity_variance with extreme values."""
    fluid_rms_velocity = np.array(
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
        fluid_rms_velocity=fluid_rms_velocity,
        collisional_radius=collisional_radius,
        particle_inertia_time=particle_inertia_time,
        particle_velocity=particle_velocity,
        taylor_microscale=0.05,
        eulerian_integral_length=1.0,
        lagrangian_integral_time=0.2,
        lagrangian_taylor_microscale_time=0.1,
    )

    assert np.all(np.isfinite(result)), "Expected all values to be finite"
