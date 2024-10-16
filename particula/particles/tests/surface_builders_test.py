"""Test builders for surface strategies, for error and validation handling.

Correctness is tested by surface_strategies_test.py.
BuilderMixin methods are tested in abc_builder_test.py.
"""

import pytest
from particula.particles.surface_builders import (
    SurfaceStrategyMolarBuilder,
    SurfaceStrategyMassBuilder,
    SurfaceStrategyVolumeBuilder,
)


def test_build_surface_strategy_molar_dict():
    """Test building with a dictionary."""
    builder = SurfaceStrategyMolarBuilder()
    parameters = {
        "surface_tension": 0.072,
        "density": 1000,
        "molar_mass": 0.01815,
    }
    builder.set_parameters(parameters)
    assert builder.surface_tension == 0.072
    assert builder.density == 1000
    assert builder.molar_mass == 0.01815

    # build the object
    strategy = builder.build()
    assert strategy.get_name() == "SurfaceStrategyMolar"


def test_build_surface_strategy_molar_missing_parameters():
    """Test building with missing parameters."""
    builder = SurfaceStrategyMolarBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert (
        "Required parameter(s) not set: surface_tension, density, molar_mass"
        in str(excinfo.value)
    )


def test_build_surface_strategy_mass_dict():
    """Test building with a dictionary."""
    builder = SurfaceStrategyMassBuilder()
    parameters = {"surface_tension": 0.072, "density": 1000}
    builder.set_parameters(parameters)
    assert builder.surface_tension == 0.072
    assert builder.density == 1000

    # build the object
    strategy = builder.build()
    assert strategy.get_name() == "SurfaceStrategyMass"


def test_build_surface_strategy_mass_missing_parameters():
    """Test building with missing parameters."""
    builder = SurfaceStrategyMassBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert "Required parameter(s) not set: surface_tension, density" in str(
        excinfo.value
    )


def test_build_surface_strategy_volume_dict():
    """Test building with a dictionary."""
    builder = SurfaceStrategyVolumeBuilder()
    parameters = {"surface_tension": 0.072, "density": 1000}
    builder.set_parameters(parameters)
    assert builder.surface_tension == 0.072
    assert builder.density == 1000

    # build the object
    strategy = builder.build()
    assert strategy.__class__.__name__ == "SurfaceStrategyVolume"


def test_build_surface_strategy_volume_missing_parameters():
    """Test building with missing parameters."""
    builder = SurfaceStrategyVolumeBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert "Required parameter(s) not set: surface_tension, density" in str(
        excinfo.value
    )
