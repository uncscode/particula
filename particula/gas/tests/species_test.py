"""Test the gas species builder"""

# pylint: disable=##R0801

import numpy as np
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)


def test_gas_species_builder_single_species():
    """Test building a single gas species with the GasSpeciesBuilder."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(
        vapor_pressure=101325
    )
    name = "Oxygen"
    molar_mass = 0.032  # kg/mol
    condensable = False
    concentration = 1.2  # kg/m^3

    gas_species = GasSpecies(
        name=name,
        molar_mass=molar_mass,
        vapor_pressure_strategy=vapor_pressure_strategy,
        condensable=condensable,
        concentration=concentration,
    )

    assert gas_species.name == name
    assert gas_species.get_molar_mass() == molar_mass
    assert gas_species.get_condensable() is condensable
    assert gas_species.get_concentration() == concentration
    assert gas_species.get_pure_vapor_pressure(298) == 101325


def test_gas_species_builder_array_species():
    """Test building an array of gas species with the GasSpeciesBuilder."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(
        vapor_pressure=np.array([101325, 101325])
    )
    names = np.array(["Oxygen", "Nitrogen"])
    molar_masses = np.array([0.032, 0.028])  # kg/mol
    condensables = np.array([False, False])
    concentrations = np.array([1.2, 0.8])  # kg/m^3

    # Assuming the builder and GasSpecies class can handle array inputs
    # directly.
    gas_species = GasSpecies(
        name=names,
        molar_mass=molar_masses,
        vapor_pressure_strategy=vapor_pressure_strategy,
        condensable=condensables,
        concentration=concentrations,
    )

    assert np.array_equal(gas_species.name, names)
    assert np.array_equal(gas_species.get_molar_mass(), molar_masses)
    assert np.array_equal(gas_species.get_condensable(), condensables)
    assert np.array_equal(gas_species.get_concentration(), concentrations)
    # This assumes get_pure_vapor_pressure can handle and return arrays
    # correctly.
    assert np.array_equal(
        gas_species.get_pure_vapor_pressure(np.array([298, 300])),
        np.array([101325, 101325]),
    )
