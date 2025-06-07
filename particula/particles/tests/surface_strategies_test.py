"""Test the surface module."""

import numpy as np
import pytest
from particula.particles.surface_strategies import (
    SurfaceStrategyMass,
    SurfaceStrategyMolar,
    SurfaceStrategyVolume,
)


# Test MolarSurfaceStrategy
def test_molar_surface_strategy():
    """Test MolarSurfaceStrategy class."""
    surface_tension = np.array([0.072, 0.05])  # water, oil
    density = np.array([1000, 800])  # water, oil
    molar_mass = np.array([0.01815, 0.03])  # water, oil

    strategy = SurfaceStrategyMolar(surface_tension, density, molar_mass)

    # Test effective surface tension
    mass_concentration = np.array([100, 200])  # water, oil
    expected_st_scalar = 0.0599547511312
    expected_surface_tension = np.full_like(surface_tension, expected_st_scalar)
    np.testing.assert_allclose(
        strategy.effective_surface_tension(mass_concentration),
        expected_surface_tension,
    )

    # Test effective density
    expected_density = density
    np.testing.assert_allclose(
        strategy.effective_density(mass_concentration), expected_density
    )

    # Test kelvin_radius
    molar_mass_water = molar_mass[0]
    expected_kelvin_radius = (
        2 * expected_surface_tension * molar_mass_water
    ) / (8.314 * 298 * expected_density)
    np.testing.assert_allclose(
        strategy.kelvin_radius(molar_mass_water, mass_concentration, 298),
        expected_kelvin_radius,
        rtol=1e-3,
    )

    # Test kelvin_term
    radius = 1e-6
    expected_kelvin_term = np.exp(expected_kelvin_radius / radius)
    np.testing.assert_allclose(
        strategy.kelvin_term(radius, molar_mass_water, mass_concentration, 298).squeeze(),
        expected_kelvin_term,
    )


# Test MassSurfaceStrategy
def test_mass_surface_strategy():
    """Test MassSurfaceStrategy class."""
    surface_tension = np.array([0.072, 0.05])  # water, oil
    density = np.array([1000, 800])  # water, oil

    strategy = SurfaceStrategyMass(surface_tension, density)

    # Test effective surface tension
    mass_concentration = np.array([100, 200])  # water, oil
    expected_surface_tension = surface_tension[0] * 100 / (
        100 + 200
    ) + surface_tension[1] * 200 / (100 + 200)
    assert strategy.effective_surface_tension(
        mass_concentration
    ) == pytest.approx(expected_surface_tension)

    # Test effective density
    expected_density = density
    np.testing.assert_allclose(
        strategy.effective_density(mass_concentration), expected_density
    )

    # Test kelvin_radius
    expected_kelvin_radius = (
        2 * expected_surface_tension * 0.01815
    ) / (8.314 * 298 * expected_density)
    np.testing.assert_allclose(
        strategy.kelvin_radius(0.01815, mass_concentration, 298),
        expected_kelvin_radius,
        rtol=1e-4,
    )

    # Test kelvin_term
    radius = 1e-6
    expected_kelvin_term = np.exp(expected_kelvin_radius / radius)
    np.testing.assert_allclose(
        strategy.kelvin_term(radius, 0.01815, mass_concentration, 298).squeeze(),
        expected_kelvin_term,
        rtol=1e-4,
    )


# Test VolumeSurfaceStrategy
def test_volume_surface_strategy():
    """Test VolumeSurfaceStrategy class."""
    surface_tension = np.array([0.072, 0.05])  # water, oil
    density = np.array([1000, 800])  # water, oil

    strategy = SurfaceStrategyVolume(surface_tension, density)

    # Test effective surface tension
    mass_concentration = np.array([100, 200])  # water, oil
    expected_surface_tension = 0.05628571428571429
    assert strategy.effective_surface_tension(
        mass_concentration
    ) == pytest.approx(expected_surface_tension)

    # Test effective density
    expected_density = density
    np.testing.assert_allclose(
        strategy.effective_density(mass_concentration), expected_density
    )

    # Test kelvin_radius
    expected_kelvin_radius = (
        2 * expected_surface_tension * 0.01815
    ) / (8.314 * 298 * expected_density)
    np.testing.assert_allclose(
        strategy.kelvin_radius(0.01815, mass_concentration, 298),
        expected_kelvin_radius,
        rtol=1e-4,
    )

    # Test kelvin_term
    radius = 1e-6
    expected_kelvin_term = np.exp(expected_kelvin_radius / radius)
    np.testing.assert_allclose(
        strategy.kelvin_term(radius, 0.01815, mass_concentration, 298).squeeze(),
        expected_kelvin_term,
        rtol=1e-4,
    )


def test_molar_surface_strategy_scalar_input():
    surface_tension = 0.072          # scalar
    density = 1000                   # scalar
    molar_mass = 0.01815             # scalar
    mass_concentration = 150         # any positive value

    strat = SurfaceStrategyMolar(surface_tension, density, molar_mass)

    assert strat.effective_surface_tension(mass_concentration) == pytest.approx(
        surface_tension
    )

def test_surface_strategy_phase_index():
    """Test phase-index mixing option."""
    surface_tension = np.array([0.072, 0.05])
    density = np.array([1000, 800])
    molar_mass = np.array([0.01815, 0.03])
    phase_index = np.array([0, 1])
    mass_concentration = np.array([100, 200])

    strat = SurfaceStrategyMolar(
        surface_tension, density, molar_mass, phase_index=phase_index
    )
    expected_st = surface_tension
    np.testing.assert_allclose(
        strat.effective_surface_tension(mass_concentration), expected_st
    )

    strat_mass = SurfaceStrategyMass(
        surface_tension, density, phase_index=phase_index
    )
    np.testing.assert_allclose(
        strat_mass.effective_surface_tension(mass_concentration), surface_tension
    )

    strat_vol = SurfaceStrategyVolume(
        surface_tension, density, phase_index=phase_index
    )
    np.testing.assert_allclose(
        strat_vol.effective_surface_tension(mass_concentration), surface_tension
    )
