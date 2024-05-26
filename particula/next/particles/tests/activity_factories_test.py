"""Test for activity factories."""

import pytest
import numpy as np

from particula.next.particles.activity_factories import (
    particle_activity_strategy_factory,
)
from particula.next.particles.activity_strategies import (
    MolarIdealActivity,
    MassIdealActivity,
    KappaParameterActivity,
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
