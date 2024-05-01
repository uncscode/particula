"""Tests for the vapor pressure module."""

import numpy as np
import pytest
from particula.constants import GAS_CONSTANT
from particula.next.gas_vapor_pressure import (
    calculate_partial_pressure,
    calculate_concentration,
    vapor_pressure_factory)


def test_calculate_partial_pressure_scalar():
    """Test the calculate_partial_pressure function with scalar values."""
    # Test the function with scalar values
    concentration = 1.0  # kg/m^3
    molar_mass = 0.029  # kg/mol (approx. for air)
    temperature = 298  # K (approx. 25 degrees Celsius)
    expected_pressure = (concentration * GAS_CONSTANT.m *
                         temperature) / molar_mass
    assert calculate_partial_pressure(
        concentration,
        molar_mass,
        temperature) == pytest.approx(expected_pressure)


def test_calculate_partial_pressure_array():
    """Test the calculate_partial_pressure function with NumPy arrays."""
    # Test the function with NumPy arrays
    concentration = np.array([1.0, 2.0])  # kg/m^3
    molar_mass = np.array([0.029, 0.032])  # kg/mol (approx. for air and O2)
    temperature = np.array([298, 310])  # K
    expected_pressure = (concentration * GAS_CONSTANT.m *
                         temperature) / molar_mass
    np.testing.assert_array_almost_equal(
        calculate_partial_pressure(concentration, molar_mass, temperature),
        expected_pressure
    )


def test_calculate_concentration_scalar():
    """Test the calculate_concentration function with scalar values."""
    # Test the function with scalar values
    partial_pressure = 101325  # Pa
    molar_mass = 0.029  # kg/mol (approx. for air)
    temperature = 298  # K (approx. 25 degrees Celsius)
    expected_concentration = (partial_pressure * molar_mass) / \
        (GAS_CONSTANT.m * temperature)
    assert calculate_concentration(
        partial_pressure,
        molar_mass,
        temperature) == pytest.approx(expected_concentration)


def test_calculate_concentration_array():
    """Test the calculate_concentration function with NumPy arrays."""
    # Test the function with NumPy arrays
    partial_pressure = np.array([101325, 202650])  # Pa
    molar_mass = np.array([0.029, 0.032])  # kg/mol (approx. for air and O2)
    temperature = np.array([298, 310])  # K
    expected_concentration = (partial_pressure * molar_mass) / \
        (GAS_CONSTANT.m * temperature)
    np.testing.assert_array_almost_equal(
        calculate_concentration(partial_pressure, molar_mass, temperature),
        expected_concentration
    )


def test_constant_vapor_pressure_strategy():
    """Test the constant vapor pressure strategy."""
    constant_pressure = 101325  # Pa, standard atmospheric pressure
    strategy = vapor_pressure_factory(
        "constant", vapor_pressure=constant_pressure)
    assert strategy.pure_vapor_pressure(298) == constant_pressure


def test_antoine_vapor_pressure_strategy():
    """Test the Antoine vapor pressure strategy."""
    # Example coefficients for water
    a, b, c = 8.07131, 1730.63, 233.426
    strategy = vapor_pressure_factory("antoine", a=a, b=b, c=c)
    temperature = 100 + 273.15  # Convert 100°C to Kelvin
    expected_pressure = 10**(a - (b / (temperature - c))) * 133.322
    assert strategy.pure_vapor_pressure(
        temperature) == pytest.approx(expected_pressure)


def test_clausius_clapeyron_strategy():
    """Test the Clausius-Clapeyron vapor pressure strategy."""
    latent_heat = 2260000  # J/kg for water
    temp_initial = 373.15  # K, boiling point of water
    pressure_initial = 101325  # Pa, standard atmospheric pressure
    strategy = vapor_pressure_factory(
        "clausius_clapeyron",
        latent_heat=latent_heat,
        temperature_initial=temp_initial,
        pressure_initial=pressure_initial)
    # Testing at a higher temperature
    temperature_final = 373.15 + 10  # K
    expected_pressure = pressure_initial * \
        np.exp(latent_heat / GAS_CONSTANT.m *
               (1 / temp_initial - 1 / temperature_final))
    assert strategy.pure_vapor_pressure(
        temperature_final) == pytest.approx(expected_pressure)


def test_water_buck_strategy():
    """Test the Buck equation for water vapor pressure."""
    strategy = vapor_pressure_factory("water_buck")
    temperature = 25+273.15  # 25°C to Kelvin
    # Use Buck equation directly for expected value calculation or reference a
    # known value
    expected_pressure = 3168.531  # Example expected result in Pa
    assert strategy.pure_vapor_pressure(
        temperature) == pytest.approx(expected_pressure)
