"""Tests for the vapor pressure factories."""

import pytest
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    AntoineVaporPressureStrategy,
    ClausiusClapeyronStrategy,
    WaterBuckStrategy,
)
from particula.gas.vapor_pressure_factories import VaporPressureFactory


def test_factory_with_constant_strategy():
    """Test factory creates a ConstantVaporPressureStrategy correctly."""
    strategy = VaporPressureFactory().get_strategy(
        strategy_type="constant", parameters={"vapor_pressure": 101325}
    )
    assert isinstance(strategy, ConstantVaporPressureStrategy)


def test_factory_with_antoine_strategy():
    """Test factory creates an AntoineVaporPressureStrategy correctly."""
    parameters = {"a": 10.0, "b": 2000.0, "c": 100.0}
    strategy = VaporPressureFactory().get_strategy(
        strategy_type="antoine", parameters=parameters
    )
    assert isinstance(strategy, AntoineVaporPressureStrategy)


def test_factory_with_clausius_clapeyron_strategy():
    """Test factory creates a ClausiusClapeyronStrategy correctly."""
    parameters = {
        "latent_heat": 2260,
        "temperature_initial": 300,
        "pressure_initial": 101325,
    }
    strategy = VaporPressureFactory().get_strategy(
        strategy_type="clausius_clapeyron", parameters=parameters
    )
    assert isinstance(strategy, ClausiusClapeyronStrategy)


def test_factory_with_water_buck_strategy():
    """Test factory creates a WaterBuckStrategy correctly without
    parameters."""
    strategy = VaporPressureFactory().get_strategy(strategy_type="water_buck")
    assert isinstance(strategy, WaterBuckStrategy)


def test_factory_with_unknown_strategy():
    """Test factory raises an error for an unknown strategy."""
    with pytest.raises(ValueError) as excinfo:
        VaporPressureFactory().get_strategy(strategy_type="unknown")
    assert "Unknown strategy type: unknown" in str(excinfo.value)


def test_factory_with_incomplete_parameters():
    """Test factory raises an error when parameters are incomplete for a
    strategy."""
    parameters = {"a": 10.0, "b": 2000.0}  # Missing 'c'
    with pytest.raises(ValueError) as excinfo:
        VaporPressureFactory().get_strategy(
            strategy_type="antoine", parameters=parameters
        )
    # Assuming builders check and raise for missing params
    assert "Missing required parameter(s): c" in str(excinfo.value)
