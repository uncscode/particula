"""
Tests for the settling velocity property calculation.
"""

import numpy as np
import pytest

from particula.particles.properties.settling_velocity import (
    particle_settling_velocity,
    get_particle_settling_velocity_via_inertia,
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


def test_get_particle_settling_velocity_via_inertia_scalar():
    """
    Test get_particle_settling_velocity_via_inertia with scalar inputs.
    """
    particle_inertia_time = 0.02  # s
    gravitational_acceleration = 9.81  # m/s²
    slip_correction_factor = 1.0  # Default (no slip correction)

    expected = (
        gravitational_acceleration
        * particle_inertia_time
        * slip_correction_factor

    )
    result = get_particle_settling_velocity_via_inertia(
        particle_inertia_time,
        gravitational_acceleration,
        slip_correction_factor,
    )

    assert np.isclose(result, expected, atol=1e-10)


def test_get_particle_settling_velocity_via_inertia_array():
    """
    Test get_particle_settling_velocity_via_inertia with NumPy array inputs.
    """
    particle_inertia_time = np.array([0.02, 0.03])
    gravitational_acceleration = 9.81  # m/s²
    slip_correction_factor = np.array(
        [1.0, 1.1]
    )  # Some slip correction for small particles

    expected = (
        gravitational_acceleration
        * particle_inertia_time
        * slip_correction_factor
    )
    result = get_particle_settling_velocity_via_inertia(
        particle_inertia_time,
        gravitational_acceleration,
        slip_correction_factor,
    )

    assert np.allclose(result, expected, atol=1e-10)


def test_get_particle_settling_velocity_via_inertia_invalid():
    """
    Test that get_particle_settling_velocity_via_inertia raises errors for
    invalid inputs.
    """
    with pytest.raises(ValueError):
        get_particle_settling_velocity_via_inertia(
            -0.02, 9.81, 1.0
        )  # Negative inertia time

    with pytest.raises(ValueError):
        get_particle_settling_velocity_via_inertia(
            0.02, -9.81, 1.0
        )  # Negative gravity

    with pytest.raises(ValueError):
        get_particle_settling_velocity_via_inertia(
            0.02, 9.81, -1.0
        )  # Negative slip correction
