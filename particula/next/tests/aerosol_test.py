"""Tests for the Aerosol class."""

# to handle pytest fixture call error
# https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly
# pylint: disable=redefined-outer-name

import pytest
import numpy as np
from particula.next.aerosol import Aerosol
from particula.next.gas.species import Gas
from particula.next.particle import Particle, create_particle_strategy


@pytest.fixture
def sample_gas():
    """Fixture for creating a Gas instance for testing."""
    gas = Gas()
    gas.add_species("Oxygen", 32.0)
    gas.add_species("Nitrogen", 28.0)
    return gas


@pytest.fixture
def sample_particles():
    """Fixture for creating a Particle instance for testing."""
    strategy = create_particle_strategy('mass_based')
    return Particle(
        strategy,
        np.array([100, 200, 300], dtype=np.float64),
        np.float64(2.5),
        np.array([10, 20, 30], dtype=np.float64))


def test_initialization(sample_gas, sample_particles):
    """Test the initialization of an Aerosol object."""
    aerosol = Aerosol(sample_gas, sample_particles)
    # Test for dynamic attachment and correct initialization
    # These tests should align with your Aerosol's implementation details
    assert hasattr(aerosol, 'gas_get_mass')
    assert hasattr(aerosol, 'particle_get_mass')


def test_replace_gas(sample_gas, sample_particles):
    """Test replacing the Gas instance in an Aerosol object."""
    aerosol = Aerosol(sample_gas, sample_particles)
    new_gas = Gas()
    new_gas.add_species("H2O", 18.0)
    aerosol.replace_gas(new_gas)
    # Ensure the gas object is replaced and methods are correctly attached
    assert hasattr(aerosol, 'gas_get_mass')
    # Verify the new gas properties are accessible
    # This requires gas_get_mass to be correctly implemented to fetch mass by
    # species name


def test_replace_particle(sample_gas, sample_particles):
    """Test replacing the Particle instance in an Aerosol object."""
    aerosol = Aerosol(sample_gas, sample_particles)
    new_strategy = create_particle_strategy(
        'radii_based')  # Assuming this changes the behavior
    new_particles = Particle(
        new_strategy,
        np.array([50, 100, 150], dtype=np.float64),
        np.float64(3.0),  # Changed density for demonstration
        np.array([15, 25, 35], dtype=np.float64))  # Changed concentration for
    aerosol.replace_particle(new_particles)
    # Ensure the particle object is replaced and methods are correctly attached
    assert hasattr(aerosol, 'particle_get_mass')
    # Verify the new particle properties are accessible and correctly
    # calculated
