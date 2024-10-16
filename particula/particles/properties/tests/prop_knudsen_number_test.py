""" test knudsen number utility

    the knudsen number goes like:
        0   for larger  particles
        inf for smaller particles
"""

import pytest
import numpy as np
from particula.particles.properties import calculate_knudsen_number


def test_basic_calculation():
    """Test the basic calculation of the Knudsen number"""
    kn = calculate_knudsen_number(0.1, 0.05)
    assert kn == 2.0


def test_numpy_array_input():
    """Test when numpy arrays are provided for radius"""
    mfp = np.array([0.1])
    radius = np.array([0.05, 0.1])
    expected_results = np.array([2.0, 1.0])
    kn = calculate_knudsen_number(mfp, radius)
    np.testing.assert_array_equal(kn, expected_results)


def test_array_in_both_inputs():
    """Test when numpy arrays are provided for both radius and mfp"""
    mfp = np.array([0.1, 0.2])
    radius = np.array([0.05, 0.1, 0.8, 3])
    expected_results = mfp[np.newaxis, :] / radius[:, np.newaxis]
    kn = calculate_knudsen_number(mfp, radius)
    np.testing.assert_array_equal(kn, expected_results)


def test_zero_particle_radius():
    """Test when the particle radius is zero,
    which should raise an exception"""
    with pytest.raises(ZeroDivisionError):
        calculate_knudsen_number(0.1, 0.0)


def test_negative_inputs():
    """Test when negative inputs are provided to the function"""
    kn = calculate_knudsen_number(-0.1, -0.05)
    assert kn == 2.0


def test_invalid_type_inputs():
    """Test when invalid input types are provided to the function"""
    with pytest.raises(TypeError):
        calculate_knudsen_number("0.1", 0.05)  # Invalid string input

    with pytest.raises(TypeError):
        calculate_knudsen_number(0.1, "0.05")  # Invalid string input
