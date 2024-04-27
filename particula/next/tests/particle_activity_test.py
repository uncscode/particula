"""Tests for the particle activity module.
Replace with real values in the future."""


import pytest
import numpy as np
from particula.next.particle_activity import (
    MolarIdealActivity,
    MassIdealActivity,
    KappaParameterActivity,
    particle_activity_strategy_factory  # type: ignore
)


# Test MolarIdealActivity
def test_molar_ideal_activity_single_species():
    """Test activity calculation for a single species."""
    activity_strategy = MolarIdealActivity()
    mass_concentration = 100.0
    expected_activity = 1.0
    assert activity_strategy.activity(mass_concentration) == expected_activity


def test_molar_ideal_activity_multiple_species():
    """Test activity calculation for multiple species."""
    activity_strategy = MolarIdealActivity(
        molar_mass=np.array([1.0, 2.0, 3.0]))
    mass_concentration = np.array([100.0, 200.0, 300.0])
    expected_activity = np.array([0.33333, 0.333333, 0.333333])
    np.testing.assert_allclose(
        activity_strategy.activity(mass_concentration),
        expected_activity,
        atol=1e-4
    )


# Test MassIdealActivity
def test_mass_ideal_activity_single_species():
    """Test activity calculation for a single species."""
    activity_strategy = MassIdealActivity()
    mass_concentration = 100.0
    expected_activity = 1.0
    assert activity_strategy.activity(mass_concentration) == expected_activity


def test_mass_ideal_activity_multiple_species():
    """Test activity calculation for multiple species."""
    activity_strategy = MassIdealActivity()
    mass_concentration = np.array([100.0, 200.0, 300.0])
    expected_activity = np.array([0.16666667, 0.33333333, 0.5])
    np.testing.assert_allclose(
        activity_strategy.activity(mass_concentration),
        expected_activity,
        atol=1e-4
    )


def test_kappa_parameter_activity_multiple_species():
    """Test activity calculation for multiple species."""
    activity_strategy = KappaParameterActivity(
        kappa=np.array([0.1, 0.2, 0.3]),
        density=np.array([1000.0, 2000.0, 3000.0]),
        molar_mass=np.array([1.0, 2.0, 3.0]),
        water_index=0
    )
    mass_concentration = np.array([100.0, 200.0, 300.0])
    expected_activity = np.array([0.66666667, 0.33333333, 0.33333333])
    np.testing.assert_allclose(
        activity_strategy.activity(mass_concentration),
        expected_activity
    )


# Test particle_activity_strategy_factory
def test_particle_activity_strategy_factory_molar_ideal():
    """Test creating a molar ideal activity strategy."""
    strategy_type = "molar_ideal"
    kwargs = {"molar_mass": np.array([1.0, 2.0, 3.0])}
    activity_strategy = particle_activity_strategy_factory(
        strategy_type, **kwargs)  # type: ignore
    assert isinstance(activity_strategy, MolarIdealActivity)


def test_particle_activity_strategy_factory_mass_ideal():
    """Test creating a mass ideal activity strategy."""
    strategy_type = "mass_ideal"
    kwargs = {}
    activity_strategy = particle_activity_strategy_factory(
        strategy_type, **kwargs)  # type: ignore
    assert isinstance(activity_strategy, MassIdealActivity)


def test_particle_activity_strategy_factory_kappa():
    """Test creating a kappa parameter activity strategy."""
    strategy_type = "kappa"
    kwargs = {
        "kappa": np.array([0.1, 0.2, 0.3]),
        "density": np.array([1000.0, 2000.0, 3000.0]),
        "water_index": 0
    }
    activity_strategy = particle_activity_strategy_factory(
        strategy_type, **kwargs)  # type: ignore
    assert isinstance(activity_strategy, KappaParameterActivity)


def test_particle_activity_strategy_factory_unknown_type():
    """Test creating a particle activity strategy with an unknown type."""
    strategy_type = "unknown_type"
    kwargs = {}
    with pytest.raises(ValueError):
        particle_activity_strategy_factory(
            strategy_type, **kwargs)  # type: ignore
