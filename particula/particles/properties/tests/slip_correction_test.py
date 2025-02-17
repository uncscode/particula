""" test the slip correction factor calculation
"""

import numpy as np
import pytest
from particula.particles.properties import cunningham_slip_correction


def test_cunningham_slip_correction_basic():
    """Test with basis input values"""
    # previous method
    knudsen_number = 0.5
    expected = 1 + knudsen_number * (
        1.257 + 0.4 * np.exp(-1.1 / knudsen_number)
    )
    assert np.isclose(cunningham_slip_correction(knudsen_number), expected)


def test_cunningham_slip_correction_array():
    """Test with an array of Knudsen numbers"""
    knudsen_numbers = np.array([0.1, 1, 10])
    expected = 1 + knudsen_numbers * (
        1.257 + 0.4 * np.exp(-1.1 / knudsen_numbers)
    )
    np.testing.assert_allclose(
        cunningham_slip_correction(knudsen_numbers), expected
    )


def test_cunningham_slip_correction_high_value():
    """Test with a high Knudsen number to check behavior approaching"""
    knudsen_number = 100
    expected = 1 + knudsen_number * (
        1.257 + 0.4 * np.exp(-1.1 / knudsen_number)
    )
    assert np.isclose(cunningham_slip_correction(knudsen_number), expected)


def test_cunningham_slip_correction_low_value():
    """Test with a low Knudsen number (approaching zero)"""
    knudsen_number = 0.01
    expected = 1 + knudsen_number * (
        1.257 + 0.4 * np.exp(-1.1 / knudsen_number)
    )
    assert np.isclose(cunningham_slip_correction(knudsen_number), expected)


def test_cunningham_slip_correction_negative_value():
    """Test with a negative Knudsen number to see if the function handles it,
    remove once value error handling is implemented in the function"""
    knudsen_number = -0.5
    expected = 1 + knudsen_number * (
        1.257 + 0.4 * np.exp(-1.1 / knudsen_number)
    )
    assert np.isclose(cunningham_slip_correction(knudsen_number), expected)


def test_cunningham_slip_correction_type_error():
    """Test with an incorrect type for the Knudsen number input"""
    knudsen_number = "0.5"  # Incorrect type (string)
    with pytest.raises(TypeError):
        cunningham_slip_correction(knudsen_number)
