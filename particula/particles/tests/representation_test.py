"""Test Building ParticleRepresentation properties."""

from typing import Any

import numpy as np

from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.distribution_strategies import (
    ParticleResolvedSpeciatedMass,
    RadiiBasedMovingBin,
)
from particula.particles.representation import ParticleRepresentation
from particula.particles.surface_strategies import SurfaceStrategyVolume


# pylint: disable=too-many-arguments, too-many-positional-arguments
def setup_particle(
    strategy: Any = RadiiBasedMovingBin(),
    activity: ActivityIdealMass = ActivityIdealMass(),
    surface: SurfaceStrategyVolume = SurfaceStrategyVolume(),
    distribution: np.ndarray = np.array([1.0, 2.0, 3.0]),
    density: Any = 1.0,
    concentration: np.ndarray = np.array([10, 20, 30]),
    charge: Any = 1.0,
) -> ParticleRepresentation:
    """Setup ParticleRepresentation for testing with configurable parameters.

    - strategy : Strategy for particle distribution
    - activity : Activity strategy for particles
    - surface : Surface strategy for particles
    - distribution : Distribution array for particles
    - density : Density of particles
    - concentration : Concentration array for particles
    - charge : Charge of particles
    """
    return ParticleRepresentation(
        strategy=strategy,
        activity=activity,
        surface=surface,
        distribution=distribution,
        density=np.atleast_1d(np.asarray(density, dtype=np.float64)),
        concentration=concentration,
        charge=np.atleast_1d(np.asarray(charge, dtype=np.float64)),
    )


def setup_particle_resolved() -> ParticleRepresentation:
    """Setup ParticleRepresentation for testing with particle resolved."""
    masses = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )

    return setup_particle(
        strategy=ParticleResolvedSpeciatedMass(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=masses,
        density=np.array([1.0, 2.0, 3.0]),
        concentration=np.array([1, 1, 1, 1]),
        charge=1.0,
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
    mass = particle.get_mass()
    expected_mass_concentration = np.sum(mass * particle.get_concentration())
    total_mass = particle.get_mass_concentration()
    total_mass_clone = particle.get_mass_concentration(clone=True)

    assert isinstance(total_mass, np.float64)
    assert isinstance(total_mass_clone, np.float64)
    assert total_mass == total_mass_clone
    assert total_mass == expected_mass_concentration


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
    conc = particle.get_concentration()
    expected_total_concentration = np.sum(conc)
    total_concentration = particle.get_total_concentration()
    total_concentration_clone = particle.get_total_concentration(clone=True)

    assert total_concentration == total_concentration_clone
    assert total_concentration == expected_total_concentration


def test_get_effective_density():
    """Test get_effective_density method."""
    particle = setup_particle()
    effective_density = particle.get_effective_density()

    assert isinstance(effective_density, np.ndarray)

    distribution = particle.get_distribution()
    expected_density = np.ones_like(distribution) * particle.get_density()
    np.testing.assert_allclose(effective_density, expected_density, rtol=1e-7)


def test_get_mean_effective_density():
    """Test get_mean_effective_density for single-species."""
    particle = setup_particle()
    med = particle.get_mean_effective_density()
    effective_density = particle.get_effective_density()
    # Filter out any zero entries before computing mean
    effective_density_nonzero = effective_density[effective_density != 0]
    expected_mean = np.mean(effective_density_nonzero)

    assert isinstance(med, float)
    assert np.isclose(med, expected_mean, rtol=1e-7)


def test_get_mean_effective_density_particle_resolved():
    """Test get_mean_effective_density for multi-species (particle-resolved)."""
    particle = setup_particle_resolved()
    med = particle.get_mean_effective_density()
    expected_mean = np.float64(2.1526515151515153)
    assert isinstance(med, float)
    assert np.isclose(med, expected_mean, rtol=1e-7)


def test_get_effective_density_particle_resolved():
    """Test get_effective_density method for particle resolved."""
    particle = setup_particle_resolved()
    effective_density = particle.get_effective_density()

    assert isinstance(effective_density, np.ndarray)

    total_mass = particle.get_mass()
    density = particle.get_density()
    # Weighted-average density for each particle (row)
    expected_density = (
        np.sum(particle.get_species_mass() * density, axis=1) / total_mass
    )
    np.testing.assert_allclose(effective_density, expected_density, rtol=1e-7)


def test_bin_order_after_add_mass():
    """Bins should be ordered after mass addition."""
    particle = setup_particle(
        distribution=np.array([2.0, 1.0, 3.0]),
        concentration=np.array([20, 10, 30]),
        charge=np.array([2, 1, 3]),
    )

    particle.add_mass(np.zeros_like(particle.get_distribution()))

    np.testing.assert_allclose(
        particle.get_distribution(), np.array([1.0, 2.0, 3.0]), rtol=1e-12
    )
    np.testing.assert_array_equal(
        particle.get_concentration(), np.array([10, 20, 30])
    )
    np.testing.assert_array_equal(particle.get_charge(), np.array([1, 2, 3]))


def test_add_concentration_charge_passthrough_updates_representation():
    """add_concentration should wire charge through the strategy."""
    particle = setup_particle(
        strategy=ParticleResolvedSpeciatedMass(),
        distribution=np.array([[1.0, 2.0]], dtype=np.float64),
        density=np.array([1.0, 1.0], dtype=np.float64),
        concentration=np.array([1.0], dtype=np.float64),
        charge=0.5,
    )

    particle.add_concentration(
        added_concentration=np.array([1.0], dtype=np.float64),
        added_distribution=np.array([[3.0, 4.0]], dtype=np.float64),
        added_charge=np.array([2.0], dtype=np.float64),
    )

    np.testing.assert_array_equal(
        particle.get_charge(), np.array([0.5, 2.0], dtype=np.float64)
    )


def test_add_concentration_charge_none_returns_none():
    """When charge is None, add_concentration should preserve None."""
    particle = ParticleRepresentation(
        strategy=ParticleResolvedSpeciatedMass(),
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=np.array([[1.0, 2.0]], dtype=np.float64),
        density=np.array([1.0, 1.0], dtype=np.float64),
        concentration=np.array([1.0], dtype=np.float64),
        charge=None,  # type: ignore[arg-type]
    )

    particle.add_concentration(
        added_concentration=np.array([1.0], dtype=np.float64),
        added_distribution=np.array([[3.0, 4.0]], dtype=np.float64),
    )

    assert particle.get_charge() is None
