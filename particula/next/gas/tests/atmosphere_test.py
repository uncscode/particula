"""Testing Gas class"""

# pylint: disable=R0801

import pytest
import numpy as np
from particula.next.gas.atmosphere import Atmosphere, AtmosphereBuilder
from particula.next.gas.species import GasSpeciesBuilder
from particula.next.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy)


def test_gas_initialization():
    """Test the initialization of a Gas object."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(
        vapor_pressure=np.array([101325, 101325]))
    names = np.array(["Oxygen1", "Nitrogen2"])
    molar_masses = np.array([0.032, 0.028])  # kg/mol
    condensables = np.array([False, False])
    concentrations = np.array([1.2, 0.8])  # kg/m^3

    gas_species = (GasSpeciesBuilder()
                   .name(names)
                   .molar_mass(molar_masses)
                   .vapor_pressure_strategy(vapor_pressure_strategy)
                   .condensable(condensables)
                   .concentration(concentrations)
                   .build())

    temperature = 298.15  # Kelvin
    total_pressure = 101325  # Pascals

    gas = Atmosphere(temperature, total_pressure, [gas_species])

    assert gas.temperature == temperature
    assert gas.total_pressure == total_pressure

    # test add_species
    assert len(gas.species) == 1

    # Add a species
    gas.add_species(gas_species=gas_species)
    assert len(gas.species) == 2

    # Remove a species
    gas.remove_species(1)
    assert len(gas.species) == 1


def test_gas_builder_with_species():
    """Test building a Gas object with the GasBuilder."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(
        vapor_pressure=np.array([101325, 101325]))
    names = np.array(["Oxygen", "Nitrogen"])
    molar_masses = np.array([0.032, 0.028])  # kg/mol
    condensables = np.array([False, False])
    concentrations = np.array([1.2, 0.8])  # kg/m^3

    gas_species = (GasSpeciesBuilder()
                   .name(names)
                   .molar_mass(molar_masses)
                   .vapor_pressure_strategy(vapor_pressure_strategy)
                   .condensable(condensables)
                   .concentration(concentrations)
                   .build())

    gas = (AtmosphereBuilder()
           .temperature(298.15)
           .total_pressure(101325)
           .add_species(gas_species)
           .build())

    assert gas.temperature == 298.15
    assert gas.total_pressure == 101325
    assert len(gas.species) == 1


def test_gas_builder_without_species_raises_error():
    """Test that building a Gas object without any species raises an error."""
    builder = AtmosphereBuilder()
    # Omit adding any species to trigger the validation error
    with pytest.raises(ValueError) as e:
        builder.build()
    assert "At least one GasSpecies must be added." in str(e.value)
