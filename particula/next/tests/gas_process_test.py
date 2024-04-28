"""Tests for gas_process"""

# pylint: disable=redefined-outer-name
# pylint: disable=R0801

import pytest
import numpy as np

from particula.next.particle import Particle, particle_strategy_factory
from particula.next.gas_vapor_pressure import ConstantVaporPressureStrategy
from particula.next.gas_species import GasSpeciesBuilder
from particula.next.gas import GasBuilder
from particula.next.aerosol import Aerosol
from particula.next.gas_process import adiabatic_pressure_change, \
    AdiabaticPressureChange
from particula.next.particle_activity import MassIdealActivity
from particula.next.surface import surface_strategy_factory

activity_strategy = MassIdealActivity()
surface = surface_strategy_factory()


def test_adiabatic_pressure_change():
    """Test the adiabatic_pressure_change function."""
    # Example conditions: Air compressed adiabatically
    t_initial = 300  # kelvin
    p_initial = 101325  # pascals (1 atm)
    p_final = 202650  # pascals (2 atm)
    gamma = 1.4  # Approx for air

    t_final_expected = t_initial * (p_final / p_initial)**((gamma - 1) / gamma)
    t_final_calculated = adiabatic_pressure_change(
        t_initial, p_initial, p_final, gamma)

    assert np.isclose(
        t_final_expected, t_final_calculated)


@pytest.fixture
def sample_gas():
    """Fixture for creating a Gas instance for testing."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(
        vapor_pressure=np.array([101325, 101326]))
    names = np.array(["Oxygen", "N2"])
    molar_masses = np.array([0.032, 0.028])  # kg/mol
    condensables = np.array([False, False])
    concentrations = np.array([1.2, 0.9])  # kg/m^3

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
    strategy = particle_strategy_factory('mass_based_moving_bin')
    return Particle(
        strategy,
        activity_strategy,
        surface,
        np.array([100, 200, 350], dtype=np.float64),
        np.float64([2.5]),
        np.array([10, 20, 50], dtype=np.float64))


@pytest.fixture
def aerosol_with_fixtures(sample_gas, sample_particles) -> Aerosol:
    """Fixture for creating an Aerosol instance with fixtures."""
    return Aerosol(gas=sample_gas, particles=sample_particles)


def test_adiabaticpressurechange(aerosol_with_fixtures):
    """Test the AdiabaticPressureChange process."""
    # Process parameters
    new_pressure = 202650.0  # Final pressure (2 atm)
    runnable = AdiabaticPressureChange(new_pressure)

    # Execute the process
    runnable.execute(aerosol_with_fixtures)

    # Expected final temperature
    t_final_expected = 298.15 * (new_pressure / 101325) ** ((1.4 - 1) / 1.4)

    # Check the final temperature
    final_temperature = aerosol_with_fixtures.gas.temperature
    assert np.isclose(final_temperature, t_final_expected)
