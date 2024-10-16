"""Test for calling the surface builders with factory.
Is the factory function working as expected, creating the right surface
strategy based on the input parameters, and raising the right exceptions
when needed.

The Builder is tested independently."""

import pytest
import numpy as np
from particula.particles.surface_factories import (
    SurfaceFactory,
)
from particula.particles.surface_strategies import (
    SurfaceStrategyMolar,
    SurfaceStrategyMass,
    SurfaceStrategyVolume,
)


def test_surface_molar_with_parameters():
    """Test factory function for surface molar strategy with parameters."""
    parameters = {
        "molar_mass": np.array([100.0, 200.0, 300.0]),
        "density": np.array([1000.0, 2000.0, 3000.0]),
        "surface_tension": np.array([0.1, 0.2, 0.3]),
    }
    strategy = SurfaceFactory().get_strategy("molar", parameters)
    assert isinstance(strategy, SurfaceStrategyMolar)
    np.testing.assert_allclose(
        strategy.molar_mass, parameters["molar_mass"], atol=1e-4
    )
    np.testing.assert_allclose(
        strategy.density, parameters["density"], atol=1e-4
    )
    np.testing.assert_allclose(
        strategy.surface_tension, parameters["surface_tension"], atol=1e-4
    )


def test_surface_mass_with_parameters():
    """Test factory function for surface mass strategy with parameters."""
    parameters = {
        "density": np.array([1000.0, 2000.0, 3000.0]),
        "surface_tension": np.array([0.1, 0.2, 0.3]),
    }
    strategy = SurfaceFactory().get_strategy("mass", parameters)
    assert isinstance(strategy, SurfaceStrategyMass)
    np.testing.assert_allclose(
        strategy.density, parameters["density"], atol=1e-4
    )
    np.testing.assert_allclose(
        strategy.surface_tension, parameters["surface_tension"], atol=1e-4
    )


def test_surface_volume_with_parameters():
    """Test factory function for surface volume strategy with parameters."""
    parameters = {
        "density": np.array([1000.0, 2000.0, 3000.0]),
        "surface_tension": np.array([0.1, 0.2, 0.3]),
    }
    strategy = SurfaceFactory().get_strategy("volume", parameters)
    assert isinstance(strategy, SurfaceStrategyVolume)
    np.testing.assert_allclose(
        strategy.density, parameters["density"], atol=1e-4
    )
    np.testing.assert_allclose(
        strategy.surface_tension, parameters["surface_tension"], atol=1e-4
    )


def test_invalid_strategy_type():
    """Test factory function with an invalid strategy type."""
    with pytest.raises(ValueError) as excinfo:
        SurfaceFactory().get_strategy("invalid_type")
    assert "Unknown strategy type: invalid_type" in str(excinfo.value)
