"""Test the calculate_concentration function."""

import numpy as np
import pytest
from particula.constants import GAS_CONSTANT

from particula.gas.properties.concentration_function import (
    calculate_concentration,
)


def test_calculate_concentration_scalar():
    """Test the calculate_concentration function with scalar values."""
    # Test the function with scalar values
    partial_pressure = 101325  # Pa
    molar_mass = 0.029  # kg/mol (approx. for air)
    temperature = 298  # K (approx. 25 degrees Celsius)
    expected_concentration = (partial_pressure * molar_mass) / (
        GAS_CONSTANT.m * temperature
    )
    assert calculate_concentration(
        partial_pressure, molar_mass, temperature
    ) == pytest.approx(expected_concentration)


def test_calculate_concentration_array():
    """Test the calculate_concentration function with NumPy arrays."""
    # Test the function with NumPy arrays
    partial_pressure = np.array([101325, 202650])  # Pa
    molar_mass = np.array([0.029, 0.032])  # kg/mol (approx. for air and O2)
    temperature = np.array([298, 310])  # K
    expected_concentration = (partial_pressure * molar_mass) / (
        GAS_CONSTANT.m * temperature
    )
    np.testing.assert_array_almost_equal(
        calculate_concentration(partial_pressure, molar_mass, temperature),
        expected_concentration,
    )
