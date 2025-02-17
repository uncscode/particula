"""Distribution Strategies testing."""

import numpy as np
import pytest
from particula.particles.distribution_strategies import (
    MassBasedMovingBin,
    RadiiBasedMovingBin,
    SpeciatedMassMovingBin,
    ParticleResolvedSpeciatedMass,
)
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.particles.activity_strategies import ActivityIdealMass


mass_based_strategy = MassBasedMovingBin()
radii_based_strategy = RadiiBasedMovingBin()
speciated_mass_strategy = SpeciatedMassMovingBin()
surface_strategy = SurfaceStrategyVolume()
activity_strategy = ActivityIdealMass()
particle_resolved_mass_strategy = ParticleResolvedSpeciatedMass()


def test_mass_based_strategy_mass():
    """Test mass calculation"""
    distribution = np.array([100, 200, 300], dtype=np.float64)
    # Example densities for different particles
    density = np.array([1, 2, 3], dtype=np.float64)
    # For MassBasedStrategy, the mass is the distribution itself
    expected_mass = distribution
    np.testing.assert_array_equal(
        mass_based_strategy.get_mass(distribution, density), expected_mass
    )


def test_mass_based_strategy_radius():
    """Test radius calculation."""
    # Example setup; replace with actual calculation for expected values
    distribution = np.array([100, 200, 300], dtype=np.float64)
    density = np.array([1, 2, 3], dtype=np.float64)
    expected_radii = (3 * distribution / (4 * np.pi * density)) ** (1 / 3)
    np.testing.assert_array_almost_equal(
        mass_based_strategy.get_radius(distribution, density), expected_radii
    )


def test_mass_based_strategy_total_mass():
    """Test total mass calculation."""
    distribution = np.array([100, 200, 300], dtype=np.float64)
    # Example concentrations for different particles
    concentration = np.array([1, 2, 3], dtype=np.float64)
    density = np.array([1, 2, 3], dtype=np.float64)  # Example densities
    expected_total_mass = np.sum(distribution * concentration)
    assert (
        mass_based_strategy.get_total_mass(
            distribution, concentration, density
        )
        == expected_total_mass
    )


def test_mass_based_strategy_add_mass():
    """Test mass addition."""
    distribution = np.array([100, 200, 300], dtype=np.float64)
    concentration = np.array([1, 2, 3], dtype=np.float64)
    density = np.array([1, 2, 3], dtype=np.float64)
    added_mass = np.array([10, 20, 30], dtype=np.float64)
    expected_distribution = distribution + added_mass
    expected_concentration = concentration
    np.testing.assert_array_equal(
        mass_based_strategy.add_mass(
            distribution, concentration, density, added_mass
        ),
        (expected_distribution, expected_concentration),
    )


def test_mass_based_strategy_collide():
    """Test mass-based strategy collision."""
    # should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        mass_based_strategy.collide_pairs(
            np.array([100, 200, 300], dtype=np.float64),
            np.array([1, 2, 3], dtype=np.float64),
            np.array([100, 200, 300], dtype=np.float64),
            np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int64),
        )


def test_radii_based_get_mass():
    """Test number-based strategy mass calculation."""
    distribution = np.array([1, 2, 3], dtype=np.float64)  # Example radii
    density = np.float64(5)  # Example density
    expected_volumes = 4 / 3 * np.pi * distribution**3
    expected_mass = expected_volumes * density
    np.testing.assert_array_almost_equal(
        radii_based_strategy.get_mass(distribution, density), expected_mass
    )


def test_radii_based_get_radius():
    """Test number-based strategy radius calculation."""
    distribution = np.array([1, 2, 3], dtype=np.float64)  # Example radii
    # Density, not used in get_radius, provided for consistency
    density = np.float64(5)
    np.testing.assert_array_equal(
        radii_based_strategy.get_radius(distribution, density), distribution
    )


def test_radii_based_get_total_mass():
    """Test number-based strategy total mass calculation."""
    distribution = np.array([1, 2, 3], dtype=np.float64)  # Example radii
    density = np.float64(5)  # Example density
    concentration = np.array([10, 20, 30], dtype=np.float64)  # concentrations
    expected_masses = 4 / 3 * np.pi * distribution**3 * density
    expected_total_mass = np.sum(expected_masses * concentration)
    assert radii_based_strategy.get_total_mass(
        distribution, concentration, density
    ) == pytest.approx(expected_total_mass)


def test_radii_based_add_mass():
    """Test number-based strategy mass addition."""
    distribution = np.array(
        [1, 2, 3], dtype=np.float64
    )  # Example radii in meters
    concentration = np.array(
        [10, 20, 30], dtype=np.float64
    )  # Concentrations in number of particles
    density = np.array(
        [2, 5, 1], dtype=np.float64
    )  # Example densities in kg/m^3
    added_mass = np.array([10, 20, 30], dtype=np.float64)  # Added mass in kg

    # Step 1: Calculate mass per particle
    mass_per_particle = (
        added_mass / concentration
    )  # Mass added to each particle

    # Step 2: Calculate new volumes (V = 4/3 * pi * r^3 for initial volume)
    # Then add the mass contribution by mass_per_particle / density
    new_volumes = (
        4 / 3
    ) * np.pi * distribution**3 + mass_per_particle / density

    # Step 3: Convert new volumes back to radii (r = (3V / 4pi)^(1/3))
    new_radii = (3 * new_volumes / (4 * np.pi)) ** (1 / 3)

    # Step 4: Set expected values (radii should increase)
    expected_distribution = new_radii
    expected_concentration = concentration  # Concentration remains the same
    np.testing.assert_array_equal(
        radii_based_strategy.add_mass(
            distribution, concentration, density, added_mass
        ),
        (expected_distribution, expected_concentration),
    )


def test_radii_based_collide():
    """Test radii-based strategy collision."""
    # should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        radii_based_strategy.collide_pairs(
            np.array([1, 2, 3], dtype=np.float64),
            np.array([5], dtype=np.float64),
            np.array([1, 2, 3], dtype=np.float64),
            np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int64),
        )


def test_speciated_mass_strategy_get_mass():
    """Test speciated mass strategy mass calculation"""
    # Example 2D distribution matrix
    distribution = np.array([[100, 200], [300, 400]], dtype=np.float64)
    # Example densities for each species
    densities = np.array([2, 3], dtype=np.float64)
    concentration = np.ones(len(densities), dtype=np.float64)
    expected_species_mass = distribution
    np.testing.assert_array_almost_equal(
        speciated_mass_strategy.get_species_mass(distribution, densities),
        expected_species_mass,
    )
    expected_mass = np.sum(distribution, axis=1)
    np.testing.assert_array_almost_equal(
        speciated_mass_strategy.get_mass(distribution, densities),
        expected_mass,
    )
    expected_total_mass = np.sum(expected_mass)
    np.testing.assert_array_almost_equal(
        speciated_mass_strategy.get_total_mass(
            distribution, concentration, densities
        ),
        expected_total_mass,
    )


def test_speciated_mass_strategy_get_radius():
    """Test speciated mass strategy radius calculation."""
    # Example 2D distribution matrix
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    # Example densities for each species
    densities = np.array([2, 3], dtype=np.float64)
    # Calculate expected volumes and then radii
    volumes = np.sum(distribution / densities, axis=1)  # Volume calculation
    expected_radii = (3 * volumes / (4 * np.pi)) ** (
        1 / 3
    )  # Radius calculation
    result = speciated_mass_strategy.get_radius(distribution, densities)
    np.testing.assert_array_almost_equal(result, expected_radii)


def test_speciated_mass_strategy_get_total_mass():
    """Test speciated mass strategy total mass calculation."""
    # Example 2D distribution matrix
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    # Example densities for each species
    densities = np.array([2, 3], dtype=np.float64)
    # Example concentrations for each species
    concentration = np.array([10, 20, 50], dtype=np.float64)
    # Expected total mass calculation
    mass_per_particles = np.sum(distribution, axis=1)
    expected_total_mass = np.sum(
        mass_per_particles * concentration
    )  # Total mass calculation
    assert speciated_mass_strategy.get_total_mass(
        distribution, concentration, densities
    ) == pytest.approx(expected_total_mass)


def test_speciated_mass_strategy_add_mass():
    """Test speciated mass strategy mass addition."""
    # Example 2D distribution matrix (mass per bin)
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    # Particle densities (not directly used here)
    densities = np.array([2, 3], dtype=np.float64)
    # Concentration (number of particles per bin)
    concentration = np.array([10, 20, 30], dtype=np.float64)
    # Total added mass per bin
    added_mass = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float64)

    # Calculate the mass added per particle
    concentration_expand = concentration[:, np.newaxis]  # Expand for 2D array
    mass_per_particle = np.where(
        concentration_expand > 0, added_mass / concentration_expand, 0
    )

    # Calculate the expected distribution (mass per particle to current mass)
    expected_distribution = np.maximum(distribution + mass_per_particle, 0)
    # Expected concentration remains unchanged
    expected_concentration = concentration
    # Call the method being tested
    result = speciated_mass_strategy.add_mass(
        distribution, concentration, densities, added_mass
    )

    # Assert the resulting distribution matches the expected distribution
    np.testing.assert_array_almost_equal(result[0], expected_distribution)
    # Assert the concentration remains unchanged
    np.testing.assert_array_almost_equal(result[1], expected_concentration)


def test_speciated_mass_strategy_collide():
    """Raise NotImplementedError for speciated mass strategy collision."""
    with pytest.raises(NotImplementedError):
        speciated_mass_strategy.collide_pairs(
            np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float64),
            np.array([2, 3, 5], dtype=np.float64),
            np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float64),
            np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int64),
        )


def test_particle_resolved_mass_strategy_get_mass():
    """Test particle-resolved mass strategy mass calculation."""
    # Example 2D distribution matrix
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    # Example densities for each species
    densities = np.array([2, 3], dtype=np.float64)
    expected_mass = np.sum(distribution, axis=1)
    np.testing.assert_array_almost_equal(
        particle_resolved_mass_strategy.get_mass(distribution, densities),
        expected_mass,
    )


def test_particle_resolved_mass_strategy_get_radius():
    """Test particle-resolved mass strategy radius calculation."""
    # Example 2D distribution matrix
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    # Example densities for each species
    densities = np.array([2, 3], dtype=np.float64)
    # Calculate expected volumes and then radii
    volumes = np.sum(distribution / densities, axis=1)  # Volume calculation
    expected_radii = (3 * volumes / (4 * np.pi)) ** (
        1 / 3
    )  # Radius calculation
    result = particle_resolved_mass_strategy.get_radius(
        distribution, densities
    )
    np.testing.assert_array_almost_equal(result, expected_radii)


def test_particle_resolved_mass_strategy_get_total_mass():
    """Test particle-resolved mass strategy total mass calculation"""
    # Example 2D distribution matrix
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    # Example densities for each species
    densities = np.array([2, 3], dtype=np.float64)
    # Example concentrations for each species
    concentration = np.array([10, 20, 50], dtype=np.float64)
    # Expected total mass calculation
    mass_per_particles = np.sum(distribution, axis=1)
    expected_total_mass = np.sum(
        mass_per_particles * concentration
    )  # Total mass calculation
    assert particle_resolved_mass_strategy.get_total_mass(
        distribution, concentration, densities
    ) == pytest.approx(expected_total_mass)


def test_particle_resolved_mass_strategy_add_mass():
    """Test particle-resolved mass strategy mass addition."""
    # Example 2D distribution matrix (mass per bin)
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600]], dtype=np.float64
    )
    # Particle densities (not directly used here)
    densities = np.array([2, 3], dtype=np.float64)
    # Concentration (number of particles per bin)
    concentration = np.array([10, 20, 30], dtype=np.float64)
    # Total added mass per bin
    added_mass = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float64)

    # Calculate the mass added per particle
    concentration_expand = concentration[:, np.newaxis]  # Expand for 2D array
    mass_per_particle = np.where(
        concentration_expand > 0, added_mass / concentration_expand, 0
    )

    # Calculate the expected distribution (mass per particle to current mass)
    expected_distribution = np.maximum(distribution + mass_per_particle, 0)
    # Expected concentration remains unchanged
    expected_concentration = concentration
    # Call the method being tested
    result = particle_resolved_mass_strategy.add_mass(
        distribution, concentration, densities, added_mass
    )

    # Assert the resulting distribution matches the expected distribution
    np.testing.assert_array_almost_equal(result[0], expected_distribution)
    # Assert the concentration remains unchanged
    np.testing.assert_array_almost_equal(result[1], expected_concentration)


def test_particle_resolved_mass_strategy_collide():
    """Test particle-resolved mass strategy collision."""
    # Example 2D distribution matrix (mass per particle)
    distribution = np.array(
        [[100, 200], [300, 400], [500, 600], [700, 800]], dtype=np.float64
    )

    # Example densities for each species
    densities = np.array([2, 3], dtype=np.float64)

    # Example concentrations for each species
    concentration = np.array([1, 1, 1, 1], dtype=np.float64)

    # Collision pairs (index pairs indicating which particles collide)
    collision_pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)

    # Expected values
    small_index = collision_pairs[:, 0]
    large_index = collision_pairs[:, 1]

    # Step 1: Create a copy of the distribution for the expected result
    expected_mass = distribution.copy()

    # Calculate the summed masses per pair and update the large index
    expected_mass[large_index, :] = (
        expected_mass[small_index, :] + expected_mass[large_index, :]
    )

    # Zero out the mass of the small_index
    expected_mass[small_index, :] = 0
    # Step 2: Update concentration
    expected_concentration = concentration.copy()
    expected_concentration[small_index] = 0

    # Call the method being tested
    result_mass, result_concentration = (
        particle_resolved_mass_strategy.collide_pairs(
            distribution.copy(),
            concentration.copy(),
            densities,
            collision_pairs,
        )
    )

    # Assert the resulting mass matches the expected distribution
    np.testing.assert_array_almost_equal(result_mass, expected_mass)
    # Assert the resulting concentration matches the expected concentration
    np.testing.assert_array_almost_equal(
        result_concentration, expected_concentration
    )
