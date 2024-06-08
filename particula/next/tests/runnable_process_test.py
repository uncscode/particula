"""Test the Process class."""

# to handle pytest fixture call error
# https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly
# pylint: disable=redefined-outer-name

import pytest
import numpy as np

from particula.next.aerosol import Aerosol
from particula.next.gas.species import Gas
from particula.next.particle import Particle, create_particle_strategy
from particula.next.process import (
    MassCondensation, MassCoagulation, ProcessSequence)


@pytest.fixture
def aerosol():
    """Fixture for creating an Aerosol instance for testing."""
    # Setup a basic Aerosol with mock Gas and Particle for testing
    gas = Gas()  # Assuming a basic constructor for Gas
    gas.add_species("Oxygen", 32.0)
    gas.add_species("Nitrogen", 28.0)

    # Particle setup
    strategy = create_particle_strategy('mass_based')
    distribution = np.array([100, 200, 300], dtype=np.float64)
    density = np.float64(2.5)
    concentration = np.array([10, 20, 30], dtype=np.float64)
    # Adjust as per your Particle constructor
    particle = Particle(strategy, distribution, density, concentration)
    return Aerosol(gas, particle)


@pytest.fixture
def mass_condensation():
    """Fixture for creating a MassCondensation process for testing."""
    return MassCondensation(other_settings="Some settings")


@pytest.fixture
def mass_coagulation():
    """Fixture for creating a MassCoagulation process for testing."""
    return MassCoagulation(other_setting2="Some other settings")


def test_mass_condensation_execute(aerosol, mass_condensation):
    """Test the MassCondensation process execute method."""
    original_distribution = aerosol.particle.distribution.copy()
    modified_aerosol = mass_condensation.execute(aerosol)
    np.testing.assert_array_equal(
        modified_aerosol.particle.distribution,
        original_distribution * 1.5)


def test_mass_coagulation_execute(aerosol, mass_coagulation):
    """Test the MassCoagulation process execute method."""
    original_distribution = aerosol.particle.distribution.copy()
    modified_aerosol = mass_coagulation.execute(aerosol)
    np.testing.assert_array_equal(
        modified_aerosol.particle.distribution,
        original_distribution * 0.5)


def test_process_sequence(aerosol, mass_condensation, mass_coagulation):
    """Test the ProcessSequence class."""
    sequence = ProcessSequence()
    sequence.add_process(mass_condensation)
    sequence.add_process(mass_coagulation)

    # Execute sequence
    modified_aerosol = sequence.execute(aerosol)

    # Verify final distribution is as expected after both processes
    # First condensation (1.5x) then coagulation (0.5x), effectively 0.75x
    # original
    expected_distribution = np.array([75., 150., 225.])
    np.testing.assert_array_equal(
        modified_aerosol.particle.distribution,
        expected_distribution)


def test_process_sequence_chaining(
        aerosol, mass_condensation, mass_coagulation):
    """Test the ProcessSequence class chaining with the | operator."""
    sequence = mass_condensation | mass_coagulation
    modified_aerosol = sequence.execute(aerosol)

    # Verify the chaining works the same as the sequential add_process calls
    expected_distribution = np.array([75., 150., 225.])
    np.testing.assert_array_equal(
        modified_aerosol.particle.distribution,
        expected_distribution)


def test_mass_condensation_rate(aerosol, mass_condensation):
    """Test the MassCondensation process rate method."""
    assert mass_condensation.rate(aerosol) == 0.5


def test_mass_coagulation_rate(aerosol, mass_coagulation):
    """Test the MassCoagulation process rate method."""
    assert mass_coagulation.rate(aerosol) == 0.5
