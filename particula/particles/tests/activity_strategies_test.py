"""Tests for the particle activity module.
Replace with real values in the future."""

import numpy as np
from particula.particles.activity_strategies import (
    ActivityIdealMass,
    ActivityIdealMolar,
    ActivityKappaParameter,
)


# Test MolarIdealActivity
def test_molar_ideal_activity_single_species():
    """Test activity calculation for a single species."""
    activity_strategy = ActivityIdealMolar()
    mass_concentration = 100.0
    expected_activity = 1.0
    assert activity_strategy.activity(mass_concentration) == expected_activity


def test_molar_ideal_activity_multiple_species():
    """Test activity calculation for multiple species."""
    activity_strategy = ActivityIdealMolar(
        molar_mass=np.array([1.0, 2.0, 3.0])
    )
    mass_concentration = np.array([100.0, 200.0, 300.0])
    expected_activity = np.array([0.33333, 0.333333, 0.333333])
    np.testing.assert_allclose(
        activity_strategy.activity(mass_concentration),
        expected_activity,
        atol=1e-4,
    )


# Test MassIdealActivity
def test_mass_ideal_activity_single_species():
    """Test activity calculation for a single species."""
    activity_strategy = ActivityIdealMass()
    mass_concentration = 100.0
    expected_activity = 1.0
    assert activity_strategy.activity(mass_concentration) == expected_activity


def test_mass_ideal_activity_multiple_species():
    """Test activity calculation for multiple species."""
    activity_strategy = ActivityIdealMass()
    mass_concentration = np.array([100.0, 200.0, 300.0])
    expected_activity = np.array([0.16666667, 0.33333333, 0.5])
    np.testing.assert_allclose(
        activity_strategy.activity(mass_concentration),
        expected_activity,
        atol=1e-4,
    )


def test_kappa_parameter_activity_multiple_species():
    """Test activity calculation for multiple species."""
    activity_strategy = ActivityKappaParameter(
        kappa=np.array([0.1, 0.2, 0.3]),
        density=np.array([1000.0, 2000.0, 3000.0]),
        molar_mass=np.array([1.0, 2.0, 3.0]),
        water_index=0,
    )
    mass_concentration = np.array([100.0, 200.0, 300.0])
    expected_activity = np.array([0.66666667, 0.33333333, 0.33333333])
    result = activity_strategy.activity(mass_concentration)
    np.testing.assert_allclose(result, expected_activity, rtol=1e-6)


def test_kappa_parameter_activity_multi_particles():
    """Test activity calculation for multiple particles."""
    activity_strategy = ActivityKappaParameter(
        kappa=np.array([0.0, 0.5]),
        density=np.array([1000.0, 2000.0]),
        molar_mass=np.array([18.0e-3, 200.0e-3]),
        water_index=0,
    )
    mass_concentration = np.array(
        [[100.0, 100.0], [500.0, 100.0], [100.0, 500.0]]
    )
    expected_activity = np.array(
        [[0.8, 0.08256881], [0.95238095, 0.01768173], [0.44444444, 0.31034483]]
    )
    result = activity_strategy.activity(mass_concentration)
    np.testing.assert_allclose(result, expected_activity, rtol=1e-6)
