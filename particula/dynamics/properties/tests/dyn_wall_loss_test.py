"""
Test the wall loss functions.
"""

import numpy as np

from particula.dynamics.properties.wall_loss_coefficient import (
    spherical_wall_loss_coefficient,
    rectangle_wall_loss_coefficient,
    rectangle_wall_loss_coefficient_via_system_state,
    spherical_wall_loss_coefficient_via_system_state,
)


def test_spherical_wall_loss_coefficient():
    """test the spherical wall loss coefficient function"""
    # Test case with hypothetical values
    _ktp_value = 1.5  # Example value
    _diffusion_coefficient_value = 0.01  # Example value
    _settling_velocity_value = 0.05  # Example value
    _chamber_radius = 2.0  # Example value

    expected_output = 0.11816  # This is a hypothetical value.

    # Call the function with the test case
    calculated_output = spherical_wall_loss_coefficient(
        _ktp_value,
        _diffusion_coefficient_value,
        _settling_velocity_value,
        _chamber_radius,
    )

    # Assertion to check if the calculated output matches the expected output
    assert np.isclose(calculated_output, expected_output, rtol=1e-4)


def test_rectangle_wall_loss():
    """test the rectangular wall loss coefficient function"""
    # Test case with hypothetical values
    ktp_value = 1.5  # Example value
    diffusion_coefficient_value = 0.01  # Example value
    settling_velocity_value = 0.05  # Example value
    dimension = (1.0, 1.0, 1.0)  # Example value

    expected_output = 0.47312  # This is a hypothetical value.

    # Call the function with the test case
    calculated_output = rectangle_wall_loss_coefficient(
        ktp_value,
        diffusion_coefficient_value,
        settling_velocity_value,
        dimension,
    )

    # Assertion to check if the calculated output matches the expected output
    assert np.isclose(calculated_output, expected_output, rtol=1e-4)


def test_spherical_system_state():
    """Test the spherical wall loss coefficient function via system state"""
    # inpust
    wall_eddy = 1.5  # Example value
    radii1 = np.array([1e-9, 1e-6, 1e-3])  # Example value
    densities = np.array([1000, 1000, 1000])  # Example value
    temperature = 298.15  # Example value
    pressure = 101325  # Example value
    chamber_radius = 2.0  # Example value

    result = spherical_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=wall_eddy,
        particle_radius=radii1,
        particle_density=densities,
        temperature=temperature,
        pressure=pressure,
        chamber_radius=chamber_radius,
    )

    expected_output = np.array([1.34039497e-03, 4.83464971e-05, 4.44868085e01])
    assert np.allclose(result, expected_output, rtol=1e-4)


def test_rectangle_system_state():
    """Test the rectangular wall loss coefficient function via system state"""
    # inpust
    wall_eddy = 1.5  # Example value
    radii2 = np.array([1e-9, 1e-6, 1e-3])  # Example value
    densities = np.array([1000, 1000, 1000])  # Example value
    temperature = 298.15  # Example value
    pressure = 101325  # Example value
    dimensions = (1.0, 1.0, 1.0)  # Example value

    result = rectangle_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=wall_eddy,
        particle_radius=radii2,
        particle_density=densities,
        temperature=temperature,
        pressure=pressure,
        chamber_dimensions=dimensions,
    )

    expected_output = np.array([5.36695219e-03, 1.39727287e-04, 1.18631490e02])
    assert np.allclose(result, expected_output, rtol=1e-4)
