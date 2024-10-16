"""Test dilution input types and values."""

import pytest
import numpy as np
from particula.dynamics.dilution import (
    volume_dilution_coefficient,
    dilution_rate,
)


# Test cases for volume_dilution_coefficient
@pytest.mark.parametrize(
    "volume, flow_rate, expected",
    [
        (1.0, 2.0, 2.0),  # Simple case with scalar inputs
        (10.0, 5.0, 0.5),  # Scalar case where flow_rate < volume
        (
            np.array([1.0, 2.0]),
            np.array([2.0, 4.0]),
            np.array([2.0, 2.0]),
        ),  # Array input case
        (10.0, 0.0, 0.0),  # Case with zero flow rate
    ],
)
def test_volume_dilution_coefficient(volume, flow_rate, expected):
    """Test the volume dilution coefficient function."""
    result = volume_dilution_coefficient(volume, flow_rate)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "volume, flow_rate",
    [
        ("1.0", 2.0),  # Non-numeric input (string)
        (None, 2.0),  # None input
        (1.0, "2.0"),  # Non-numeric flow_rate
        (np.array([1.0, 2.0]), "5"),  # Mismatched types (array vs scalar)
    ],
)
def test_volume_dilution_coefficient_invalid_input(volume, flow_rate):
    """Test the volume dilution coefficient function with invalid inputs."""
    with pytest.raises(TypeError):
        volume_dilution_coefficient(volume, flow_rate)


# Test cases for dilution_rate
@pytest.mark.parametrize(
    "coefficient, concentration, expected",
    [
        (2.0, 3.0, -6.0),  # Simple scalar case
        (0.5, 10.0, -5.0),  # Scalar case with smaller coefficient
        (
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([-3.0, -8.0]),
        ),  # Array input case
        (1.0, 0.0, 0.0),  # Case with zero concentration
    ],
)
def test_dilution_rate(coefficient, concentration, expected):
    """Test the dilution rate function."""
    result = dilution_rate(coefficient, concentration)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "coefficient, concentration",
    [
        ("1.0", 2.0),  # Non-numeric input (string)
        (None, 2.0),  # None input
        (1.0, "2.0"),  # Non-numeric concentration
        (np.array([1.0, 2.0]), "5"),  # Mismatched types (array vs scalar)
    ],
)
def test_dilution_rate_invalid_input(coefficient, concentration):
    """Test the dilution rate function with invalid inputs."""
    with pytest.raises(TypeError):
        dilution_rate(coefficient, concentration)
