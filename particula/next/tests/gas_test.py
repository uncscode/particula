"""Testing Gas class"""

import pytest
import numpy as np
from particula.next.gas import Gas, GasBuilder
from particula.next.gas_species import GasSpeciesBuilder
from particula.next.vapor_pressure import ConstantVaporPressureStrategy


def test_gas_initialization():
    """Test the initialization of a Gas object."""
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

    temperature = 298.15  # Kelvin
    total_pressure = 101325  # Pascals

    gas = Gas(temperature, total_pressure, [gas_species])

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

    gas = (GasBuilder()
           .temperature(298.15)
           .total_pressure(101325)
           .add_species(gas_species)
           .build())

    assert gas.temperature == 298.15
    assert gas.total_pressure == 101325
    assert len(gas.species) == 1


def test_gas_builder_without_species_raises_error():
    """Test that building a Gas object without any species raises an error."""
    builder = GasBuilder()
    # Omit adding any species to trigger the validation error
    with pytest.raises(ValueError) as e:
        builder.build()
    assert "At least one gas component must be added." in str(e.value)
