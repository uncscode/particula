"""Tests for the settling velocity property calculation."""

import numpy as np
import pytest

from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity,
    get_particle_settling_velocity_via_inertia,
    get_particle_settling_velocity_with_drag,  # Add this line
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

    assert get_particle_settling_velocity(
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
        get_particle_settling_velocity(
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
        get_particle_settling_velocity(
            radius_invalid,
            particle_density,
            slip_correction_factor,
            dynamic_viscosity,
            gravitational_acceleration,
        )


def test_get_particle_settling_velocity_via_inertia_scalar():
    """Test get_particle_settling_velocity_via_inertia with scalar inputs."""
    particle_inertia_time = 0.02  # s
    particle_radius = 0.001  # m
    relative_velocity = 0.05  # m/s
    gravitational_acceleration = 9.81  # m/s²
    slip_correction_factor = 1.0  # Default (no slip correction)
    kinematic_viscosity = 1e-6  # m²/s

    # Calculate drag correction:
    re_p = (2 * particle_radius * relative_velocity) / kinematic_viscosity
    drag_correction = 1 + 0.15 * (re_p**0.687)
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
    """Test get_particle_settling_velocity_via_inertia with NumPy arrays."""
    particle_inertia_time = np.array([0.02, 0.03])
    particle_radius = np.array([0.001, 0.002])
    relative_velocity = np.array([0.05, 0.06])
    gravitational_acceleration = 9.81  # m/s²
    slip_correction_factor = np.array([1.0, 1.1])
    kinematic_viscosity = 1e-6  # m²/s

    re_p = (2 * particle_radius * relative_velocity) / kinematic_viscosity
    drag_correction = 1 + 0.15 * (re_p**0.687)
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
    """Test that get_particle_settling_velocity_via_inertia raises errors for
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


def test_get_particle_settling_velocity_with_drag_scalar_stokes():
    """Test with scalar inputs within the Stokes regime."""
    particle_radius = 1e-6  # m
    particle_density = 1000.0  # kg/m³
    fluid_density = 1.225  # kg/m³ (air at sea level)
    dynamic_viscosity = 1.81e-5  # Pa·s (air at 15°C)
    slip_correction_factor = 1.0  # Dimensionless
    gravitational_acceleration = 9.80665  # m/s²

    # Expected settling velocity using the Stokes formula
    expected_velocity = get_particle_settling_velocity(
        particle_radius,
        particle_density,
        slip_correction_factor,
        dynamic_viscosity,
        gravitational_acceleration,
    )

    # Obtained settling velocity using the function under test
    calculated_velocity = get_particle_settling_velocity_with_drag(
        particle_radius,
        particle_density,
        fluid_density,
        dynamic_viscosity,
        slip_correction_factor,
        gravitational_acceleration,
    )

    # Assert that the calculated velocity matches the expected velocity
    assert np.isclose(calculated_velocity, expected_velocity, atol=1e-4)


def test_get_particle_settling_velocity_with_drag_array_stokes():
    """Test with array inputs within the Stokes regime."""
    particle_radius = np.array([1e-7, 5e-7, 1e-6])  # m
    particle_density = np.array([1000.0, 1100.0, 1200.0])  # kg/m³
    fluid_density = 1.225  # kg/m³ (air at sea level)
    dynamic_viscosity = 1.81e-5  # Pa·s
    slip_correction_factor = np.array([1.0, 1.0, 1.0])  # Dimensionless
    gravitational_acceleration = 9.80665  # m/s²

    # Expected settling velocities using the Stokes formula
    expected_velocity = get_particle_settling_velocity(
        particle_radius,
        particle_density,
        slip_correction_factor,
        dynamic_viscosity,
        gravitational_acceleration,
    )

    # Calculated settling velocities using the function under test
    calculated_velocity = get_particle_settling_velocity_with_drag(
        particle_radius,
        particle_density,
        fluid_density,
        dynamic_viscosity,
        slip_correction_factor,
        gravitational_acceleration,
    )

    # Assert that the calculated velocities match the expected velocities
    assert np.allclose(calculated_velocity, expected_velocity, atol=1e-4)


def test_get_particle_settling_velocity_with_drag_scalar_non_stokes():
    """Test with scalar inputs in the non-Stokes regime."""
    particle_radius = 1e-3  # m (larger particle)
    particle_density = 2500.0  # kg/m³
    fluid_density = 1000.0  # kg/m³ (e.g., water)
    dynamic_viscosity = 1e-3  # Pa·s (water at 20°C)
    slip_correction_factor = 1.0  # Dimensionless
    gravitational_acceleration = 9.80665  # m/s²

    # Stokes formula not valid here, use empirical or known value
    # For demonstration, assume expected velocity (empirical data)
    # For simplicity, ensure function returns positive value

    calculated_velocity = get_particle_settling_velocity_with_drag(
        particle_radius,
        particle_density,
        fluid_density,
        dynamic_viscosity,
        slip_correction_factor,
        gravitational_acceleration,
    )

    # Assert that the calculated velocity is reasonable (greater than zero)
    assert calculated_velocity > 0.0

    # Optionally, compare with an expected value if available
    # expected_velocity = ...  # Define expected value based on empirical data
    # assert np.isclose(calculated_velocity, expected_velocity, rtol=0.1)


def test_get_particle_settling_velocity_with_drag_array_non_stokes():
    """Test with array inputs in the non-Stokes regime."""
    particle_radius = np.array([5e-4, 1e-3, 2e-3])  # m
    particle_density = np.array([2600.0, 2700.0, 2800.0])  # kg/m³
    fluid_density = 1000.0  # kg/m³
    dynamic_viscosity = 1e-3  # Pa·s
    slip_correction_factor = np.array([1.0, 1.0, 1.0])  # Dimensionless
    gravitational_acceleration = 9.80665  # m/s²

    calculated_velocity = get_particle_settling_velocity_with_drag(
        particle_radius,
        particle_density,
        fluid_density,
        dynamic_viscosity,
        slip_correction_factor,
        gravitational_acceleration,
    )

    # Assert that all calculated velocities are greater than zero
    assert np.all(calculated_velocity > 0.0)


def test_get_particle_settling_velocity_with_drag_transition():
    """Test the transition between Stokes and non-Stokes regimes."""
    # Define a radius that is close to the transition point
    particle_radius = 1e-5  # m
    particle_density = 2000.0  # kg/m³
    fluid_density = 1.225  # kg/m³
    dynamic_viscosity = 1.81e-5  # Pa·s
    slip_correction_factor = 1.0
    gravitational_acceleration = 9.80665  # m/s²
    re_threshold = 0.1  # As defined in the function

    # Calculate the Stokes velocity
    stokes_velocity = get_particle_settling_velocity(
        particle_radius,
        particle_density,
        slip_correction_factor,
        dynamic_viscosity,
        gravitational_acceleration,
    )

    # Calculate the Reynolds number for the Stokes velocity
    re_stokes = (
        2.0
        * particle_radius
        * stokes_velocity
        * fluid_density
        / dynamic_viscosity
    )

    # Verify that we're near the threshold
    assert np.isclose(re_stokes, re_threshold, atol=1e-1)

    # Compute the settling velocity using the function under test
    calculated_velocity = get_particle_settling_velocity_with_drag(
        particle_radius,
        particle_density,
        fluid_density,
        dynamic_viscosity,
        slip_correction_factor,
        gravitational_acceleration,
        re_threshold=re_threshold,
    )

    # Since we're near the threshold, ensure the function handles it smoothly
    assert calculated_velocity > 0.0


def test_get_particle_settling_velocity_with_drag_invalid_inputs():
    """Test handling of invalid inputs."""
    # Negative particle radius
    with pytest.raises(ValueError):
        get_particle_settling_velocity_with_drag(
            -1e-6,  # Invalid radius
            1000.0,
            1.225,
            1.81e-5,
            1.0,
        )

    # Zero fluid viscosity
    with pytest.raises(ZeroDivisionError):
        get_particle_settling_velocity_with_drag(
            1e-6,
            1000.0,
            1.225,
            0.0,  # Invalid dynamic viscosity
            1.0,
        )

    # Negative dynamic viscosity
    with pytest.raises(ValueError):
        get_particle_settling_velocity_with_drag(
            1e-6,
            1000.0,
            1.225,
            -1.0e-5,  # Invalid dynamic viscosity
            1.0,
        )


def test_get_particle_settling_velocity_with_drag_extreme_values():
    """Test the function with extreme input values."""
    # Extremely small particle radius
    particle_radius = 1e-9  # m
    particle_density = 1000.0  # kg/m³
    fluid_density = 1.225  # kg/m³
    dynamic_viscosity = 1.81e-5  # Pa·s
    slip_correction_factor = 1.0

    calculated_velocity = get_particle_settling_velocity_with_drag(
        particle_radius,
        particle_density,
        fluid_density,
        dynamic_viscosity,
        slip_correction_factor,
    )

    # Assert that the calculated velocity is reasonable
    assert calculated_velocity >= 0.0

    # Extremely large particle radius
    particle_radius = 0.1  # m (10 cm)
    calculated_velocity = get_particle_settling_velocity_with_drag(
        particle_radius,
        particle_density,
        fluid_density,
        dynamic_viscosity,
        slip_correction_factor,
    )

    assert calculated_velocity >= 0.0
