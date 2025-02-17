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
    expected_surface_tension = 0.0599547511312
    assert strategy.effective_surface_tension(
        mass_concentration
    ) == pytest.approx(expected_surface_tension)

    # Test effective density
    expected_density = 890.497737556561
    assert strategy.effective_density(mass_concentration) == pytest.approx(
        expected_density
    )

    # Test kelvin_radius
    molar_mass_water = molar_mass[0]
    expected_kelvin_radius = (
        2 * expected_surface_tension * molar_mass_water
    ) / (8.314 * 298 * expected_density)
    assert strategy.kelvin_radius(
        molar_mass_water, mass_concentration, 298
    ) == pytest.approx(expected_kelvin_radius, rel=1e-3)

    # Test kelvin_term
    radius = 1e-6
    expected_kelvin_term = np.exp(expected_kelvin_radius / radius)
    assert strategy.kelvin_term(
        radius, molar_mass_water, mass_concentration, 298
    ) == pytest.approx(expected_kelvin_term)


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
    expected_density = density[0] * 100 / (100 + 200) + density[1] * 200 / (
        100 + 200
    )
    assert strategy.effective_density(mass_concentration) == pytest.approx(
        expected_density
    )

    # Test kelvin_radius
    expected_kelvin_radius = 9.691952451532837e-10
    assert strategy.kelvin_radius(
        0.01815, mass_concentration, 298
    ) == pytest.approx(expected_kelvin_radius, rel=1e-6)

    # Test kelvin_term
    radius = 1e-6
    expected_kelvin_term = np.exp(expected_kelvin_radius / radius)
    assert strategy.kelvin_term(
        radius, 0.01815, mass_concentration, 298
    ) == pytest.approx(expected_kelvin_term, rel=1e-6)


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
    expected_density = 857.1428571428572
    assert strategy.effective_density(mass_concentration) == pytest.approx(
        expected_density
    )

    # Test kelvin_radius
    expected_kelvin_radius = 9.62057760789752e-10
    assert strategy.kelvin_radius(
        0.01815, mass_concentration, 298
    ) == pytest.approx(expected_kelvin_radius)

    # Test kelvin_term
    radius = 1e-6
    expected_kelvin_term = np.exp(expected_kelvin_radius / radius)
    assert strategy.kelvin_term(
        radius, 0.01815, mass_concentration, 298
    ) == pytest.approx(expected_kelvin_term)
