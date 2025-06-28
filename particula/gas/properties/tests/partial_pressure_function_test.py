"""Tests for the calculate_partial_pressure function."""

import numpy as np
import pytest

from particula.gas.properties.pressure_function import (
    get_partial_pressure,
)
from particula.util.constants import GAS_CONSTANT


def test_calculate_partial_pressure_scalar():
    """Test the calculate_partial_pressure function with scalar values."""
    # Test the function with scalar values
    concentration = 1.0  # kg/m^3
    molar_mass = 0.029  # kg/mol (approx. for air)
    temperature = 298  # K (approx. 25 degrees Celsius)
    expected_pressure = (
        concentration * GAS_CONSTANT * temperature
    ) / molar_mass
    assert get_partial_pressure(
        concentration, molar_mass, temperature
    ) == pytest.approx(expected_pressure)


def test_calculate_partial_pressure_array():
    """Test the calculate_partial_pressure function with NumPy arrays."""
    # Test the function with NumPy arrays
    concentration = np.array([1.0, 2.0])  # kg/m^3
    molar_mass = np.array([0.029, 0.032])  # kg/mol (approx. for air and O2)
    temperature = np.array([298, 310])  # K
    expected_pressure = (
        concentration * GAS_CONSTANT * temperature
    ) / molar_mass
    np.testing.assert_array_almost_equal(
        get_partial_pressure(concentration, molar_mass, temperature),
        expected_pressure,
    )


def test_calculate_partial_pressure_edge_cases():
    """Test the calculate_partial_pressure function with edge case values."""
    test_cases = [
        (1e-6, 0.029, 298),  # Very low concentration
        (1e6, 0.029, 298),  # Very high concentration
        (1.0, 0.001, 298),  # Very low molar mass
        (1.0, 1.0, 298),  # High molar mass
        (1.0, 0.029, 1),  # Very low temperature
        (1.0, 0.029, 1e4),  # Very high temperature
    ]
    for concentration, molar_mass, temperature in test_cases:
        expected_pressure = (
            concentration * GAS_CONSTANT * temperature
        ) / molar_mass
        assert get_partial_pressure(
            concentration, molar_mass, temperature
        ) == pytest.approx(expected_pressure)
