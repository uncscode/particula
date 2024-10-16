"""Test the gas species builder, for building gas species objects.
and checking/validation of the parameters.
"""

import pytest
from particula.gas.species import GasSpecies
from particula.gas.species_builders import (
    GasSpeciesBuilder,
    PresetGasSpeciesBuilder,
)
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)


def test_set_name():
    """Test setting the name of the gas species."""
    builder = GasSpeciesBuilder()
    builder.set_name("Oxygen")
    assert builder.name == "Oxygen"


def test_set_molar_mass():
    """Test setting the molar mass of the gas species."""
    builder = GasSpeciesBuilder()
    builder.set_molar_mass(0.032)
    assert builder.molar_mass == 0.032

    with pytest.raises(ValueError):
        builder.set_molar_mass(-0.032)


def test_set_vapor_pressure_strategy():
    """Test setting the vapor pressure strategy of the gas species."""
    builder = GasSpeciesBuilder()
    strategy = ConstantVaporPressureStrategy(0.0)
    builder.set_vapor_pressure_strategy(strategy)
    assert builder.vapor_pressure_strategy == strategy


def test_set_condensable():
    """Test setting the condensable bool of the gas species."""
    builder = GasSpeciesBuilder()
    builder.set_condensable(True)
    assert builder.condensable is True

    builder.set_condensable(False)
    assert builder.condensable is False


def test_set_concentration():
    """Test setting the concentration of the gas species."""
    builder = GasSpeciesBuilder()
    builder.set_concentration(1.0)
    assert builder.concentration == 1.0

    with pytest.raises(ValueError):
        builder.set_concentration(-1.0)

    builder.set_concentration(1.0, concentration_units="g/m^3")
    assert builder.concentration == 1.0e-3


def test_set_parameters():
    """Test setting all parameters at once."""
    builder = GasSpeciesBuilder()
    vapor_obj = ConstantVaporPressureStrategy(0.0)
    parameters = {
        "name": "Oxygen",
        "molar_mass": 0.032,
        "vapor_pressure_strategy": vapor_obj,
        "condensable": True,
        "concentration": 1.0,
    }
    builder.set_parameters(parameters)
    assert builder.name == "Oxygen"
    assert builder.molar_mass == 0.032
    assert builder.vapor_pressure_strategy == vapor_obj
    assert builder.condensable is True
    assert builder.concentration == 1.0


def test_missing_required_parameter():
    """Test missing a required parameter in the parameters dict."""
    builder = GasSpeciesBuilder()
    parameters = {
        "name": "Oxygen",
        "molar_mass": 0.032,
        "vapor_pressure_strategy": ConstantVaporPressureStrategy(0.0),
        "condensable": True,
    }
    with pytest.raises(ValueError):
        builder.set_parameters(parameters)


def test_invalid_parameter():
    """Test an invalid parameter in the parameters dict."""
    builder = GasSpeciesBuilder()
    parameters = {
        "name": "Oxygen",
        "molar_mass": 0.032,
        "vapor_pressure_strategy": ConstantVaporPressureStrategy(0.0),
        "condensable": True,
        "concentration": 1.0,
        "invalid_param": 123,
    }
    with pytest.raises(ValueError):
        builder.set_parameters(parameters)


def test_build():
    """Test building the gas species object."""
    builder = GasSpeciesBuilder()
    parameters = {
        "name": "Oxygen",
        "molar_mass": 0.032,
        "vapor_pressure_strategy": ConstantVaporPressureStrategy(0.0),
        "condensable": True,
        "concentration": 1.0,
    }
    builder.set_parameters(parameters)
    gas_species = builder.build()
    assert isinstance(gas_species, GasSpecies)
    assert gas_species.get_name() == "Oxygen"


def test_preset_builder():
    """Test the preset gas species builder."""
    builder = PresetGasSpeciesBuilder()
    gas_species = builder.build()
    assert isinstance(gas_species, GasSpecies)
    assert gas_species.get_name() == "Preset100"
    assert gas_species.get_molar_mass() == 0.100
    assert gas_species.get_condensable() is False
    assert gas_species.get_concentration() == 1.0
