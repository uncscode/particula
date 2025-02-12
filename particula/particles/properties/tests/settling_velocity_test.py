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
    particle_radius = 0.001  # m
    relative_velocity = 0.05  # m/s
    gravitational_acceleration = 9.81  # m/s²
    slip_correction_factor = 1.0  # Default (no slip correction)
    kinematic_viscosity = 1e-6  # m²/s

    # Calculate drag correction:
    r_e = (2 * particle_radius * relative_velocity) / kinematic_viscosity
    drag_correction = 1 + 0.15 * (r_e**0.687)
    expected = (
        gravitational_acceleration
        * particle_inertia_time
        * slip_correction_factor
        / drag_correction
    )
    result = get_particle_settling_velocity_via_inertia(
        particle_inertia_time,
        particle_radius,
        relative_velocity,
        slip_correction_factor,
        gravitational_acceleration,
        kinematic_viscosity,
    )

    assert np.isclose(result, expected, atol=1e-10)


def test_get_particle_settling_velocity_via_inertia_array():
    """
    Test get_particle_settling_velocity_via_inertia with NumPy array inputs.
    """
    particle_inertia_time = np.array([0.02, 0.03])
    particle_radius = np.array([0.001, 0.002])
    relative_velocity = np.array([0.05, 0.06])
    gravitational_acceleration = 9.81  # m/s²
    slip_correction_factor = np.array([1.0, 1.1])
    kinematic_viscosity = 1e-6  # m²/s

    Re = (2 * particle_radius * relative_velocity) / kinematic_viscosity
    drag_correction = 1 + 0.15 * (Re**0.687)
    expected = (
        gravitational_acceleration
        * particle_inertia_time
        * slip_correction_factor
        / drag_correction
    )
    result = get_particle_settling_velocity_via_inertia(
        particle_inertia_time,
        particle_radius,
        relative_velocity,
        slip_correction_factor,
        gravitational_acceleration,
        kinematic_viscosity,
    )

    assert np.allclose(result, expected, atol=1e-10)


def test_get_particle_settling_velocity_via_inertia_invalid():
    """
    Test that get_particle_settling_velocity_via_inertia raises errors for
    invalid inputs.
    """
    kinematic_viscosity = 1e-6  # m²/s
    # Test passing an invalid type for particle_inertia_time.
    with pytest.raises(TypeError):
        get_particle_settling_velocity_via_inertia(
            "invalid", 0.001, 0.05, 1.0, 9.81, kinematic_viscosity
        )

    # Test negative particle_inertia_time.
    with pytest.raises(ValueError):
        get_particle_settling_velocity_via_inertia(
            -0.02, 0.001, 0.05, 1.0, 9.81, kinematic_viscosity
        )

    with pytest.raises(ValueError):
        get_particle_settling_velocity_via_inertia(
            0.02, 0.001, 0.05, -1.0, 9.81, kinematic_viscosity
        )

    with pytest.raises(ValueError):
        get_particle_settling_velocity_via_inertia(
            0.02, 0.001, 0.05, 1.0, -9.81, kinematic_viscosity
        )
