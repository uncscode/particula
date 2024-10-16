"""Tests for the AtmosphereBuilder class."""

import pytest
import numpy as np
from particula.gas.atmosphere_builders import AtmosphereBuilder
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)


def test_gas_builder_with_species():
    """Test building a Gas object with the GasBuilder."""
    vapor_pressure_strategy_atmo = ConstantVaporPressureStrategy(
        vapor_pressure=np.array([101325, 101325, 101325])
    )
    names_atmo = np.array(["Oxygen", "Nitrogen", "Carbon Dioxide"])
    molar_masses_atmo = np.array([0.032, 0.028, 0.044])
    condensables_atmo = np.array([False, False, True])
    concentrations_atmo = np.array([0.21, 0.79, 0.0])
    gas_species_builder_test = GasSpecies(
        name=names_atmo,
        molar_mass=molar_masses_atmo,
        vapor_pressure_strategy=vapor_pressure_strategy_atmo,
        condensable=condensables_atmo,
        concentration=concentrations_atmo,
    )
    atmo = (
        AtmosphereBuilder()
        .set_temperature(298.15)
        .set_pressure(101325)
        .add_species(gas_species_builder_test)
        .build()
    )

    assert atmo.temperature == 298.15
    assert atmo.total_pressure == 101325
    assert len(atmo.species) == 1


def test_gas_builder_without_species_raises_error():
    """Test that building a Gas object without any species raises an error."""
    builder = AtmosphereBuilder()
    # Omit adding any species to trigger the validation error
    with pytest.raises(ValueError) as e:
        builder.build()
    assert "Atmosphere must contain at least one species." in str(e.value)
