"""Test Building ParticleRepresentation properties."""

import numpy as np
from particula.particles.representation import ParticleRepresentation
from particula.particles.distribution_strategies import (
    RadiiBasedMovingBin,
)
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.particles.activity_strategies import ActivityIdealMass


def setup_particle() -> ParticleRepresentation:
    """Setup ParticleRepresentation for testing."""
    set_strategy = RadiiBasedMovingBin()
    set_activity = ActivityIdealMass()
    set_surface = SurfaceStrategyVolume()
    set_distribution = np.array([1.0, 2.0, 3.0])
    set_density = 1.0
    set_concentration = np.array([10, 20, 30])
    set_charge = 1.0

    return ParticleRepresentation(
        strategy=set_strategy,
        activity=set_activity,
        surface=set_surface,
        distribution=set_distribution,
        density=set_density,
        concentration=set_concentration,
        charge=set_charge,
    )


def test_get_mass():
    """Test get_mass method."""
    particle = setup_particle()
    mass = particle.get_mass()
    mass_clone = particle.get_mass(clone=True)

    assert isinstance(mass, np.ndarray)
    assert isinstance(mass_clone, np.ndarray)
    np.testing.assert_array_equal(mass, mass_clone)


def test_get_radius():
    """Test get_radius method."""
    particle = setup_particle()
    radius = particle.get_radius()
    radius_clone = particle.get_radius(clone=True)

    assert isinstance(radius, np.ndarray)
    assert isinstance(radius_clone, np.ndarray)
    np.testing.assert_array_equal(radius, radius_clone)


def test_get_mass_concentration():
    """Test get_mass_concentration method."""
    particle = setup_particle()
    total_mass = particle.get_mass_concentration()
    total_mass_clone = particle.get_mass_concentration(clone=True)

    assert isinstance(total_mass, np.float64)
    assert isinstance(total_mass_clone, np.float64)
    assert total_mass == total_mass_clone


def test_get_strategy():
    """Test get_strategy method."""
    particle = setup_particle()
    strategy = particle.get_strategy()
    strategy_clone = particle.get_strategy(clone=True)

    assert isinstance(strategy, RadiiBasedMovingBin)
    assert isinstance(strategy_clone, RadiiBasedMovingBin)


def test_get_activity():
    """Test get_activity method."""
    particle = setup_particle()
    activity = particle.get_activity()
    activity_clone = particle.get_activity(clone=True)

    assert isinstance(activity, ActivityIdealMass)
    assert isinstance(activity_clone, ActivityIdealMass)


def test_get_surface():
    """Test get_surface method."""
    particle = setup_particle()
    surface = particle.get_surface()
    surface_clone = particle.get_surface(clone=True)

    assert isinstance(surface, SurfaceStrategyVolume)
    assert isinstance(surface_clone, SurfaceStrategyVolume)


def test_get_distribution():
    """Test get_distribution method."""
    particle = setup_particle()
    distribution = particle.get_distribution()
    distribution_clone = particle.get_distribution(clone=True)

    np.testing.assert_array_equal(distribution, distribution_clone)


def test_get_density():
    """Test get_density method."""
    particle = setup_particle()
    density = particle.get_density()
    density_clone = particle.get_density(clone=True)

    np.testing.assert_array_equal(density, density_clone)


def test_get_concentration():
    """Test get_concentration method."""
    particle = setup_particle()
    concentration = particle.get_concentration()
    concentration_clone = particle.get_concentration(clone=True)

    np.testing.assert_array_equal(concentration, concentration_clone)


def test_get_charge():
    """Test get_charge method."""
    particle = setup_particle()
    charge = particle.get_charge()
    charge_clone = particle.get_charge(clone=True)

    np.testing.assert_array_equal(charge, charge_clone)


def test_get_volume():
    """Test get_volume method."""
    particle = setup_particle()
    volume = particle.get_volume()
    volume_clone = particle.get_volume(clone=True)

    assert volume == volume_clone


def test_get_species_mass():
    """Test get_species_mass method."""
    particle = setup_particle()
    species_mass = particle.get_species_mass()
    species_mass_clone = particle.get_species_mass(clone=True)

    np.testing.assert_array_equal(species_mass, species_mass_clone)


def test_get_total_concentration():
    """Test get_total_concentration method."""
    particle = setup_particle()
    total_concentration = particle.get_total_concentration()
    total_concentration_clone = particle.get_total_concentration(clone=True)

    assert total_concentration == total_concentration_clone
