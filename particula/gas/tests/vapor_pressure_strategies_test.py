"""Tests for the vapor pressure module."""

import numpy as np
import pytest
from particula.constants import GAS_CONSTANT
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    AntoineVaporPressureStrategy,
    ClausiusClapeyronStrategy,
    WaterBuckStrategy,
)


def test_constant_vapor_pressure_strategy():
    """Test the constant vapor pressure strategy."""
    constant_pressure = 101325  # Pa, standard atmospheric pressure
    strategy = ConstantVaporPressureStrategy(vapor_pressure=constant_pressure)
    assert strategy.pure_vapor_pressure(298) == constant_pressure


def test_antoine_vapor_pressure_strategy():
    """Test the Antoine vapor pressure strategy."""
    # Example coefficients for water
    a, b, c = 8.07131, 1730.63, 233.426
    strategy = AntoineVaporPressureStrategy(a=a, b=b, c=c)
    temperature = 100 + 273.15  # Convert 100°C to Kelvin
    expected_pressure = (
        10 ** (a - (b / (temperature - c))) * 133.32238741499998
    )
    assert strategy.pure_vapor_pressure(temperature) == pytest.approx(
        expected_pressure
    )


def test_clausius_clapeyron_strategy():
    """Test the Clausius-Clapeyron vapor pressure strategy."""
    latent_heat = 2260000  # J/kg for water
    temp_initial = 373.15  # K, boiling point of water
    pressure_initial = 101325  # Pa, standard atmospheric pressure
    strategy = ClausiusClapeyronStrategy(
        latent_heat=latent_heat,
        temperature_initial=temp_initial,
        pressure_initial=pressure_initial,
    )
    # Testing at a higher temperature
    temperature_final = 373.15 + 10  # K
    expected_pressure = pressure_initial * np.exp(
        latent_heat
        / GAS_CONSTANT.m
        * (1 / temp_initial - 1 / temperature_final)
    )
    assert strategy.pure_vapor_pressure(temperature_final) == pytest.approx(
        expected_pressure
    )


def test_water_buck_strategy():
    """Test the Buck equation for water vapor pressure."""
    strategy = WaterBuckStrategy()
    temperature = 25 + 273.15  # 25°C to Kelvin
    # Use Buck equation directly for expected value calculation or reference a
    # known value
    expected_pressure = 3168.531  # Example expected result in Pa
    assert strategy.pure_vapor_pressure(temperature) == pytest.approx(
        expected_pressure
    )
