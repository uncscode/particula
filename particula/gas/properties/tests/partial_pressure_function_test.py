"""Tests for the calculate_partial_pressure function."""

import numpy as np
import pytest
from particula.constants import GAS_CONSTANT

from particula.gas.properties.pressure_function import (
    calculate_partial_pressure,
)


def test_calculate_partial_pressure_scalar():
    """Test the calculate_partial_pressure function with scalar values."""
    # Test the function with scalar values
    concentration = 1.0  # kg/m^3
    molar_mass = 0.029  # kg/mol (approx. for air)
    temperature = 298  # K (approx. 25 degrees Celsius)
    expected_pressure = (
        concentration * GAS_CONSTANT.m * temperature
    ) / molar_mass
    assert calculate_partial_pressure(
        concentration, molar_mass, temperature
    ) == pytest.approx(expected_pressure)


def test_calculate_partial_pressure_array():
    """Test the calculate_partial_pressure function with NumPy arrays."""
    # Test the function with NumPy arrays
    concentration = np.array([1.0, 2.0])  # kg/m^3
    molar_mass = np.array([0.029, 0.032])  # kg/mol (approx. for air and O2)
    temperature = np.array([298, 310])  # K
    expected_pressure = (
        concentration * GAS_CONSTANT.m * temperature
    ) / molar_mass
    np.testing.assert_array_almost_equal(
        calculate_partial_pressure(concentration, molar_mass, temperature),
        expected_pressure,
    )
