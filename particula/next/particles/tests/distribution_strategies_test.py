"""Distribution Strategies testing."""

import numpy as np
import pytest
from particula.next.particles.distribution_strategies import (
    MassBasedMovingBin, RadiiBasedMovingBin, SpeciatedMassMovingBin,
)
from particula.next.particles.surface_strategies import SurfaceStrategyVolume
from particula.next.particles.activity_strategies import IdealActivityMass


mass_based_strategy = MassBasedMovingBin()
radii_based_strategy = RadiiBasedMovingBin()
speciated_mass_strategy = SpeciatedMassMovingBin()
surface_strategy = SurfaceStrategyVolume()
activity_strategy = IdealActivityMass()


def test_mass_based_strategy_mass():
    """Test mass calculation"""
    distribution = np.array([100, 200, 300], dtype=np.float64)
    # Example densities for different particles
    density = np.array([1, 2, 3], dtype=np.float64)
    # For MassBasedStrategy, the mass is the distribution itself
    expected_mass = distribution
    np.testing.assert_array_equal(
        mass_based_strategy.get_mass(
            distribution, density), expected_mass)


def test_mass_based_strategy_radius():
    """Test radius calculation."""
    # Example setup; replace with actual calculation for expected values
    distribution = np.array([100, 200, 300], dtype=np.float64)
    density = np.array([1, 2, 3], dtype=np.float64)
    expected_radii = (3 * distribution / (4 * np.pi * density)) ** (1 / 3)
    np.testing.assert_array_almost_equal(
        mass_based_strategy.get_radius(
            distribution, density), expected_radii)


def test_mass_based_strategy_total_mass():
    """Test total mass calculation."""
    distribution = np.array([100, 200, 300], dtype=np.float64)
    # Example concentrations for different particles
    concentration = np.array([1, 2, 3], dtype=np.float64)
    density = np.array([1, 2, 3], dtype=np.float64)  # Example densities
    expected_total_mass = np.sum(distribution * concentration)
    assert mass_based_strategy.get_total_mass(
        distribution, concentration, density) == expected_total_mass


def test_radii_based_get_mass():
    """Test number-based strategy mass calculation."""
    distribution = np.array([1, 2, 3], dtype=np.float64)  # Example radii
    density = np.float64(5)  # Example density
    expected_volumes = 4 / 3 * np.pi * distribution ** 3
    expected_mass = expected_volumes * density
    np.testing.assert_array_almost_equal(
        radii_based_strategy.get_mass(distribution, density),
        expected_mass
    )


def test_radii_based_get_radius():
    """Test number-based strategy radius calculation."""
    distribution = np.array([1, 2, 3], dtype=np.float64)  # Example radii
    # Density, not used in get_radius, provided for consistency
    density = np.float64(5)
    np.testing.assert_array_equal(
        radii_based_strategy.get_radius(distribution, density),
        distribution
    )


def test_radii_based_get_total_mass():
    """Test number-based strategy total mass calculation."""
    distribution = np.array([1, 2, 3], dtype=np.float64)  # Example radii
    density = np.float64(5)  # Example density
    concentration = np.array([10, 20, 30], dtype=np.float64)  # concentrations
    expected_masses = 4 / 3 * np.pi * distribution ** 3 * density
    expected_total_mass = np.sum(expected_masses * concentration)
    assert radii_based_strategy.get_total_mass(
        distribution, concentration, density) \
        == pytest.approx(expected_total_mass)


def test_speciated_mass_strategy_get_mass():
    """Test speciated mass strategy mass calculation"""
    # Example 2D distribution matrix
    distribution = np.array([[100, 200], [300, 400]], dtype=np.float64)
    # Example densities for each species
    densities = np.array([2, 3], dtype=np.float64)
    expected_mass = np.sum(distribution, axis=1)  # Expected mass calculation
    np.testing.assert_array_almost_equal(
        speciated_mass_strategy.get_mass(distribution, densities),
        expected_mass
    )


def test_speciated_mass_strategy_get_radius():
    """Test speciated mass strategy radius calculation."""
    # Example 2D distribution matrix
    distribution = np.array([[100, 200], [300, 400]], dtype=np.float64)
    # Example densities for each species
    densities = np.array([2, 3], dtype=np.float64)
    # Calculate expected volumes and then radii
    volumes = np.sum(distribution / densities, axis=0)  # Volume calculation
    expected_radii = (3 * volumes / (4 * np.pi)
                      ) ** (1 / 3)  # Radius calculation
    np.testing.assert_array_almost_equal(
        speciated_mass_strategy.get_radius(distribution, densities),
        expected_radii
    )


def test_speciated_mass_strategy_get_total_mass():
    """Test speciated mass strategy total mass calculation."""
    # Example 2D distribution matrix
    distribution = np.array([[100, 200], [300, 400]], dtype=np.float64)
    # Example densities for each species
    densities = np.array([2, 3], dtype=np.float64)
    # Example concentrations for each species
    concentration = np.array([10, 20], dtype=np.float64)
    # Expected total mass calculation
    mass_per_species = speciated_mass_strategy.get_mass(
        distribution, densities)
    expected_total_mass = np.sum(
        mass_per_species *
        concentration)  # Total mass calculation
    assert speciated_mass_strategy.get_total_mass(
        distribution, concentration, densities) \
        == pytest.approx(expected_total_mass)
