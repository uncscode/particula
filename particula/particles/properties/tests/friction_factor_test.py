"""Test the friction_factor function."""

import numpy as np
from particula.particles.properties import friction_factor


def test_friction_factor_scalar():
    """Test the friction_factor function with scalar inputs."""
    radius = 0.1  # meters
    dynamic_viscosity = 1.81e-5  # Pascal-second, typical air at room temp
    slip_correction = 1.0  # No slip correction

    result = friction_factor(radius, dynamic_viscosity, slip_correction)

    expected_result = 6 * np.pi * dynamic_viscosity * radius / slip_correction
    assert np.isclose(result, expected_result), "Test failed for scalar inputs"


def test_friction_factor_array():
    """Test the friction_factor function with array inputs."""
    radius = np.array([0.1, 0.2])
    dynamic_viscosity = 1.81e-5  # Pascal-second, typical air at room temp
    slip_correction = np.array([1.0, 1.5])  # Different slip corrections

    result = friction_factor(radius, dynamic_viscosity, slip_correction)

    expected_result = 6 * np.pi * dynamic_viscosity * radius / slip_correction
    assert np.allclose(result, expected_result), "Test failed for array inputs"


def test_with_zero_radius():
    """Test the friction_factor function with zero radius."""
    radius = 0.0
    dynamic_viscosity = 1.81e-5
    slip_correction = 1.0

    result = friction_factor(radius, dynamic_viscosity, slip_correction)

    assert result == 0, "Test failed with zero radius"


def test_continuum_limit():
    """Test the friction_factor function in the continuum limit."""
    # Continuum limit where slip correction factor approaches 1
    radius = 0.1  # meters
    dynamic_viscosity = 1.81e-5  # Pascal-second, typical air at room temp

    result = friction_factor(radius, dynamic_viscosity, 1.0)

    # Expected result using the continuum limit formula
    expected_result = 6 * np.pi * dynamic_viscosity * radius
    assert np.isclose(
        result, expected_result
    ), "Test failed for continuum limit"


def test_kinetic_limit():
    """Test the friction_factor function in the kinetic limit."""
    # Kinetic limit where slip correction factor is significantly higher
    radius = 10e-9  # meters
    dynamic_viscosity = 1.81e-5  # Pascal-second
    slip_correction = 10.0  # Higher slip correction factor

    result = friction_factor(radius, dynamic_viscosity, slip_correction)

    expected_result = 6 * np.pi * dynamic_viscosity * radius / slip_correction
    assert np.isclose(result, expected_result), "Test failed for kinetic limit"
