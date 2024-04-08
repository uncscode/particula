"""Tests for the Aerosol class."""

# to handle pytest fixture call error
# https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly
# pylint: disable=redefined-outer-name
# pylint: disable=similar-lines

import pytest
import numpy as np
from particula.next.aerosol import Aerosol
from particula.next.gas import Gas, GasBuilder
from particula.next.gas_species import GasSpeciesBuilder
from particula.next.vapor_pressure import ConstantVaporPressureStrategy
from particula.next.particle import Particle, create_particle_strategy


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
def aerosol_with_fixtures(sample_gas, sample_particles):
    """Fixture for creating an Aerosol instance with fixtures."""
    return Aerosol(gases=sample_gas, particles=sample_particles)


def test_add_gas(aerosol_with_fixtures, sample_gas):
    """"Test adding a Gas instance."""
    aerosol_with_fixtures.add_gas(sample_gas)
    assert len(aerosol_with_fixtures.gases) == 2


def test_add_particle(aerosol_with_fixtures, sample_particles):
    """Test adding a Particle instance."""
    aerosol_with_fixtures.add_particle(sample_particles)
    assert len(aerosol_with_fixtures.particles) == 2


def test_iterate_gases(aerosol_with_fixtures):
    """Test iterating over Gas instances."""
    for gas in aerosol_with_fixtures.iterate_gas():
        assert isinstance(gas, Gas)


def test_iterate_particles(aerosol_with_fixtures):
    """Test iterating over Particle instances."""
    for particle in aerosol_with_fixtures.iterate_particle():
        assert isinstance(particle, Particle)
