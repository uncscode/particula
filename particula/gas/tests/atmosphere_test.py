"""Testing Gas class"""

# pylint: disable=R0801

import numpy as np
from particula.gas.atmosphere import Atmosphere
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)


def test_gas_initialization():
    """Test the initialization of a Gas object."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(
        vapor_pressure=np.array([101325, 101325, 101325])
    )
    names = np.array(["Oxygen1", "Hydrogen1", "Nitrogen1"])
    molar_masses = np.array([0.032, 0.002, 0.028])
    condensables = np.array([False, False, False])
    concentrations = np.array([0.21, 0.79, 0.0])
    gas_species_atmo_test = GasSpecies(
        name=names,
        molar_mass=molar_masses,
        vapor_pressure_strategy=vapor_pressure_strategy,
        condensable=condensables,
        concentration=concentrations,
    )
    # create a gas object
    temperature = 298.15  # Kelvin
    total_pressure = 101325  # Pascals
    gas = Atmosphere(temperature, total_pressure, [gas_species_atmo_test])

    assert gas.temperature == temperature
    assert gas.total_pressure == total_pressure

    # test add_species
    assert len(gas.species) == 1

    # Add a species
    gas.add_species(gas_species=gas_species_atmo_test)
    assert len(gas.species) == 2

    # Remove a species
    gas.remove_species(1)
    assert len(gas.species) == 1
