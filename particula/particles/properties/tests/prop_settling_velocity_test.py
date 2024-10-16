"""Tests for the settling velocity property calculation."""

import numpy as np
import pytest
from particula.particles.properties.settling_velocity import (
    particle_settling_velocity,
)


def test_calculate_settling_velocity_with_floats():
    """Test the settling velocity calculation for a single float value."""
    radius = 0.001
    density = 1000.0
    slip_correction_factor = 0.9
    dynamic_viscosity = 0.001
    gravitational_acceleration = 9.80665

    expected_settling_velocity = (
        (2 * radius) ** 2
        * density
        * slip_correction_factor
        * gravitational_acceleration
        / (18 * dynamic_viscosity)
    )

    assert particle_settling_velocity(
        radius,
        density,
        slip_correction_factor,
        dynamic_viscosity,
        gravitational_acceleration,
    ) == pytest.approx(expected_settling_velocity)


def test_calculate_settling_velocity_with_np_array():
    """Test the settling velocity calculation for a numpy array."""
    radii_array = np.array([1e-9, 1e-6, 1e-3])
    density_array = np.array([1000.0, 2000.0, 3000.0])
    slip_correction_factor = np.array([0.9, 0.8, 0.7])
    dynamic_viscosity = 0.001
    gravitational_acceleration = 9.80665

    expected_settling_velocity = (
        (2 * radii_array) ** 2
        * density_array
        * slip_correction_factor
        * gravitational_acceleration
        / (18 * dynamic_viscosity)
    )

    assert np.allclose(
        particle_settling_velocity(
            radii_array,
            density_array,
            slip_correction_factor,
            dynamic_viscosity,
            gravitational_acceleration,
        ),
        expected_settling_velocity,
    )


def test_calculate_settling_velocity_with_invalid_inputs():
    """Test the settling velocity calculation with invalid inputs."""
    radius_invalid = "invalid"
    particle_density = 1000.0
    slip_correction_factor = 0.9
    dynamic_viscosity = 0.001
    gravitational_acceleration = 9.80665

    with pytest.raises(TypeError):
        particle_settling_velocity(
            radius_invalid,
            particle_density,
            slip_correction_factor,
            dynamic_viscosity,
            gravitational_acceleration,
        )
