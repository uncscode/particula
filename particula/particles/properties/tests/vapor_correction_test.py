"""Test the vapor transition correction properties."""

import numpy as np
from particula.particles.properties import vapor_transition_correction


def test_vapor_transition_correction_scalar():
    """Test the vapor_transition_correction function with scalar values."""
    knudsen_number = 0.5
    mass_accommodation = 0.1
    result = vapor_transition_correction(knudsen_number, mass_accommodation)
    expected = (0.75 * 0.1 * (1 + 0.5)) / (
        (0.5**2 + 0.5) + 0.283 * 0.1 * 0.5 + 0.75 * 0.1
    )
    assert np.isclose(result, expected), "Failed for scalar inputs"


def test_vapor_transition_correction_array():
    """Test the vapor_transition_correction function with NumPy arrays."""
    knudsen_number = np.array([0.5, 1.0, 2.0])
    mass_accommodation = np.array([0.1, 0.2, 0.3])
    result = vapor_transition_correction(knudsen_number, mass_accommodation)
    expected = (0.75 * mass_accommodation * (1 + knudsen_number)) / (
        (knudsen_number**2 + knudsen_number)
        + 0.283 * mass_accommodation * knudsen_number
        + 0.75 * mass_accommodation
    )
    assert np.allclose(result, expected), "Failed for array inputs"


def test_vapor_transition_correction_zero_knudsen():
    """Test the vapor_transition_correction function with
    zero Knudsen number."""
    knudsen_number = 0
    mass_accommodation = 0.2
    result = vapor_transition_correction(knudsen_number, mass_accommodation)
    expected = (0.75 * 0.2 * (1 + 0)) / (
        (0**2 + 0) + 0.283 * 0.2 * 0 + 0.75 * 0.2
    )
    assert np.isclose(result, expected), "Failed for zero Knudsen number"


def test_vapor_transition_correction_zero_accommodation():
    """Test the vapor_transition_correction function with
    zero mass accommodation."""
    knudsen_number = 1
    mass_accommodation = 0
    result = vapor_transition_correction(knudsen_number, mass_accommodation)
    expected = (0.75 * 0 * (1 + 1)) / ((1**2 + 1) + 0.283 * 0 * 1 + 0.75 * 0)
    assert np.isclose(result, expected), "Failed for zero mass accommodation"


def test_vapor_transition_correction_high_values():
    """Test the vapor_transition_correction function with high values."""
    knudsen_number = 100
    mass_accommodation = 1
    result = vapor_transition_correction(knudsen_number, mass_accommodation)
    expected = (0.75 * 1 * (1 + 100)) / (
        (100**2 + 100) + 0.283 * 1 * 100 + 0.75 * 1
    )
    assert np.isclose(result, expected), "Failed for high values"
