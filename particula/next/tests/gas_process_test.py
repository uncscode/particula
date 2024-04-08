"""Tests for gas_process"""

import pytest
from particula.next.particle import Particle, create_particle_strategy
from particula.next.vapor_pressure import ConstantVaporPressureStrategy
from particula.next.gas_species import GasSpeciesBuilder
from particula.next.gas import GasBuilder
from particula.next.aerosol import Aerosol
import numpy as np
from particula.next.gas_process import adiabatic_pressure_change, \
    AdiabaticPressureChange


def test_adiabatic_pressure_change():
    # Example conditions: Air compressed adiabatically
    T_initial = 300  # Kelvin
    P_initial = 101325  # Pascals (1 atm)
    P_final = 202650  # Pascals (2 atm)
    gamma = 1.4  # Approx for air

    T_final_expected = T_initial * (P_final / P_initial)**((gamma - 1) / gamma)
    T_final_calculated = adiabatic_pressure_change(
        T_initial, P_initial, P_final, gamma)

    assert np.isclose(
        T_final_expected, T_final_calculated)


@pytest.fixture
def sample_gas():
    """Fixture for creating a Gas instance for testing."""
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
    return gas


@pytest.fixture
def sample_particles():
    """Fixture for creating a Particle instance for testing."""
    strategy = create_particle_strategy('mass_based')
    return Particle(
        strategy,
        np.array([100, 200, 300], dtype=np.float64),
        np.float64([2.5]),
        np.array([10, 20, 30], dtype=np.float64))


@pytest.fixture
def aerosol_with_fixtures(sample_gas, sample_particles) -> Aerosol:
    """Fixture for creating an Aerosol instance with fixtures."""
    return Aerosol(gases=sample_gas, particles=sample_particles)


def test_AdiabaticPressureChange(aerosol_with_fixtures):
    # Process parameters
    new_pressure = 202650  # Final pressure (2 atm)
    runnable = AdiabaticPressureChange(aerosol_with_fixtures, new_pressure)

    # Execute the process
    runnable.execute(aerosol_with_fixtures)

    # Expected final temperature
    T_final_expected = 298.15 * (new_pressure / 101325) ** ((1.4 - 1) / 1.4)

    # Check the final temperature
    final_temperature = aerosol_with_fixtures.gases[0].temperature
    assert np.isclose(final_temperature, T_final_expected)
