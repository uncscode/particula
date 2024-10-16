"""Tests for the Aerosol class.

Wait for particles to be implemented before running these tests."""

# to handle pytest fixture call error
# https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly
# pylint: disable=redefined-outer-name
# pylint: disable=R0801

# import pytest
# import numpy as np
# from particula.aerosol import Aerosol
# from particula.gas.atmosphere import AtmosphereBuilder
# from particula.gas.species import GasSpeciesBuilder, GasSpecies
# from particula.gas.vapor_pressure import ConstantVaporPressureStrategy
# from particula.particles.representation import Particle, particle_strategy_factory
# from particula.particles.activity import MassIdealActivity
# from particula.particles.surface import surface_strategy_factory

# activity_strategy = MassIdealActivity()
# surface_strategy = surface_strategy_factory()


# @pytest.fixture
# def sample_gas():
#     """Fixture for creating a Gas instance for testing."""
#     vapor_pressure_strategy = ConstantVaporPressureStrategy(
#         vapor_pressure=np.array([101325, 101325]))
#     names = np.array(["Oxygen", "Nitrogen"])
#     molar_masses = np.array([0.032, 0.028])  # kg/mol
#     condensables = np.array([False, False])
#     concentrations = np.array([1.2, 0.8])  # kg/m^3

#     gas_species = (GasSpeciesBuilder()
#                    .name(names)
#                    .molar_mass(molar_masses)
#                    .vapor_pressure_strategy(vapor_pressure_strategy)
#                    .condensable(condensables)
#                    .concentration(concentrations)
#                    .build())

#     return (
#         AtmosphereBuilder()
#         .temperature(298.15)
#         .total_pressure(101325)
#         .add_species(gas_species)
#         .build()
#     )


# @pytest.fixture
# def sample_particles():
#     """Fixture for creating a Particle instance for testing."""
#     strategy = particle_strategy_factory('mass_based_moving_bin')
#     return Particle(
#         strategy,
#         activity_strategy,
#         surface_strategy,
#         np.array([100, 200, 300], dtype=np.float64),
#         np.float64([2.5]),
#         np.array([10, 20, 30], dtype=np.float64))


# @pytest.fixture
# def aerosol_with_fixtures(sample_gas, sample_particles):
#     """Fixture for creating an Aerosol instance with fixtures."""
#     return Aerosol(gas=sample_gas, particles=sample_particles)


# def test_add_particle(aerosol_with_fixtures, sample_particles):
#     """Test adding a Particle instance."""
#     aerosol_with_fixtures.add_particle(sample_particles)
#     assert len(aerosol_with_fixtures.particles) == 2


# def test_iterate_gases(aerosol_with_fixtures):
#     """Test iterating over Gas instances."""
#     for gas_species in aerosol_with_fixtures.iterate_gas():
#         assert isinstance(gas_species, GasSpecies)


# def test_iterate_particles(aerosol_with_fixtures):
#     """Test iterating over Particle instances."""
#     for particle in aerosol_with_fixtures.iterate_particle():
#         assert isinstance(particle, Particle)
