"""Tests for the latent heat factories."""

import particula
import pytest
from particula.gas import (
    ConstantLatentHeat,
    ConstantLatentHeatBuilder,
    LatentHeatFactory,
    LatentHeatStrategy,
    LinearLatentHeat,
    LinearLatentHeatBuilder,
    PowerLawLatentHeat,
    PowerLawLatentHeatBuilder,
)


def test_factory_constant():
    """Test factory creates a ConstantLatentHeat strategy correctly."""
    parameters = {
        "latent_heat_ref": 2.26e6,
        "latent_heat_ref_units": "J/kg",
    }
    strategy = LatentHeatFactory().get_strategy(
        strategy_type="constant",
        parameters=parameters,
    )
    assert isinstance(strategy, ConstantLatentHeat)


def test_factory_get_builders():
    """Test factory returns expected builder mapping."""
    builders = LatentHeatFactory().get_builders()

    assert set(builders.keys()) == {"constant", "linear", "power_law"}
    assert isinstance(builders["constant"], ConstantLatentHeatBuilder)
    assert isinstance(builders["linear"], LinearLatentHeatBuilder)
    assert isinstance(builders["power_law"], PowerLawLatentHeatBuilder)


def test_factory_linear():
    """Test factory creates a LinearLatentHeat strategy correctly."""
    parameters = {
        "latent_heat_ref": 2.26e6,
        "latent_heat_ref_units": "J/kg",
        "slope": 1200.0,
        "slope_units": "J/(kg*K)",
        "temperature_ref": 298.15,
        "temperature_ref_units": "K",
    }
    strategy = LatentHeatFactory().get_strategy(
        strategy_type="linear",
        parameters=parameters,
    )
    assert isinstance(strategy, LinearLatentHeat)


def test_factory_power_law():
    """Test factory creates a PowerLawLatentHeat strategy correctly."""
    parameters = {
        "latent_heat_ref": 2.26e6,
        "latent_heat_ref_units": "J/kg",
        "critical_temperature": 647.096,
        "critical_temperature_units": "K",
        "beta": 0.38,
    }
    strategy = LatentHeatFactory().get_strategy(
        strategy_type="power_law",
        parameters=parameters,
    )
    assert isinstance(strategy, PowerLawLatentHeat)


def test_factory_invalid_type():
    """Test factory raises an error for an unknown strategy type."""
    with pytest.raises(ValueError, match="Unknown strategy type: unknown"):
        LatentHeatFactory().get_strategy("unknown")


def test_factory_incomplete_parameters():
    """Test factory raises an error when parameters are incomplete."""
    parameters = {
        "latent_heat_ref": 2.26e6,
        "latent_heat_ref_units": "J/kg",
        "temperature_ref": 298.15,
        "temperature_ref_units": "K",
    }
    with pytest.raises(
        ValueError,
        match=r"Missing required parameter\(s\): (slope|slope_units)",
    ):
        LatentHeatFactory().get_strategy("linear", parameters)


def test_factory_round_trip():
    """Test factory returns strategy that computes the expected value."""
    parameters = {
        "latent_heat_ref": 2.26e6,
        "latent_heat_ref_units": "J/kg",
    }
    strategy = LatentHeatFactory().get_strategy("constant", parameters)
    assert strategy.latent_heat(300.0) == 2.26e6


def test_import_from_gas_module():
    """Test import of latent heat strategies from particula.gas."""
    assert ConstantLatentHeat is not None
    assert LinearLatentHeat is not None
    assert PowerLawLatentHeat is not None
    assert callable(LatentHeatFactory)


def test_import_builders_from_gas():
    """Test import of latent heat builders from particula.gas."""
    assert ConstantLatentHeatBuilder is not None
    assert LinearLatentHeatBuilder is not None
    assert PowerLawLatentHeatBuilder is not None


def test_import_abc_from_gas():
    """Test import of the latent heat strategy ABC from particula.gas."""
    assert LatentHeatStrategy is not None


def test_import_particula_gas_factory():
    """Test top-level particula import exposes the gas factory."""
    assert hasattr(particula.gas, "LatentHeatFactory")
