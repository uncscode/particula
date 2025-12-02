"""Tests for the vapor pressure module."""

import numpy as np
import pytest

from particula.gas.vapor_pressure_strategies import (
    AntoineVaporPressureStrategy,
    ArblasterLiquidVaporPressureStrategy,
    ClausiusClapeyronStrategy,
    ConstantVaporPressureStrategy,
    LiquidClausiusHybridStrategy,
    TableVaporPressureStrategy,
    WaterBuckStrategy,
)
from particula.util.constants import GAS_CONSTANT


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
    expected_pressure = 10 ** (a - (b / (temperature - c))) * 133.32238741499998
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
        latent_heat / GAS_CONSTANT * (1 / temp_initial - 1 / temperature_final)
    )
    assert strategy.pure_vapor_pressure(temperature_final) == pytest.approx(
        expected_pressure
    )


def test_water_buck_strategy():
    """Test the Buck equation for water vapor pressure."""
    strategy = WaterBuckStrategy()
    temperature = 25 + 273.15  # 25°C to Kelvin
    # Use Buck equation directly for expected value or reference value
    expected_pressure = 3168.531  # Example expected result in Pa
    assert strategy.pure_vapor_pressure(temperature) == pytest.approx(
        expected_pressure
    )


def test_arblaster_liquid_vapor_pressure_strategy():
    """Test the 5-term liquid polynomial strategy."""
    coeffs = (0.0, 0.0, 0.0, 0.0, 0.0)
    strategy = ArblasterLiquidVaporPressureStrategy(coefficients=coeffs)
    assert strategy.pure_vapor_pressure(300.0) == pytest.approx(1e5)


def test_liquid_clausius_hybrid_strategy():
    """Test hybrid strategy smoothly blends above boiling."""
    coeffs = (0.0, 0.0, 0.0, 0.0, 0.0)
    latent_heat = 4.0e4
    temp_init = 350.0
    pressure_init = 1.0e5
    strategy = LiquidClausiusHybridStrategy(
        coefficients=coeffs,
        latent_heat=latent_heat,
        temperature_initial=temp_init,
        pressure_initial=pressure_init,
        boiling_point=temp_init,
        transition_width=1.0,
    )
    test_temp = 352.0
    p_liq = 1.0e5
    p_claus = pressure_init * np.exp(
        latent_heat / GAS_CONSTANT * (1 / temp_init - 1 / test_temp)
    )
    weight = 1 / (1 + np.exp(-(test_temp - temp_init) / 1.0))
    expected = (1 - weight) * p_liq + weight * p_claus
    assert strategy.pure_vapor_pressure(test_temp) == pytest.approx(expected)


def test_table_vapor_pressure_strategy_scalar_and_array():
    """Interpolate vapor pressures from tables."""
    temps = np.array([300.0, 350.0, 400.0])
    pressures = np.array([1000.0, 2000.0, 3000.0])
    strategy = TableVaporPressureStrategy(
        vapor_pressures=pressures,
        temperatures=temps,
    )
    assert strategy.pure_vapor_pressure(325.0) == pytest.approx(1500.0)
    result = strategy.pure_vapor_pressure(np.array([325.0, 375.0]))
    assert np.allclose(result, [1500.0, 2500.0])
