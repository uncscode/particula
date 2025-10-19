"""Test builders for surface strategies, for error and validation handling.

Correctness is tested by surface_strategies_test.py.
BuilderMixin methods are tested in abc_builder_test.py.
"""

import numpy as np
import pytest

from particula.particles.surface_builders import (
    SurfaceStrategyMassBuilder,
    SurfaceStrategyMolarBuilder,
    SurfaceStrategyVolumeBuilder,
)


def test_build_surface_strategy_molar_dict():
    """Test building with a dictionary."""
    builder = SurfaceStrategyMolarBuilder()
    parameters = {
        "surface_tension": 0.072,
        "surface_tension_units": "N/m",
        "density": 1000,
        "density_units": "kg/m^3",
        "molar_mass": 0.01815,
        "molar_mass_units": "kg/mol",
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
    expected_msg = "Required parameter(s) not set: "
    expected_msg += "surface_tension, density, molar_mass"
    assert expected_msg in str(excinfo.value)


def test_build_surface_strategy_mass_dict():
    """Test building with a dictionary."""
    builder = SurfaceStrategyMassBuilder()
    parameters = {
        "surface_tension": 0.072,
        "surface_tension_units": "N/m",
        "density": 1000,
        "density_units": "kg/m^3",
    }
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
    parameters = {
        "surface_tension": 0.072,
        "surface_tension_units": "N/m",
        "density": 1000,
        "density_units": "kg/m^3",
    }
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


def test_build_surface_strategy_phase_index():
    """Test optional phase index parameter."""
    builder = SurfaceStrategyMassBuilder()
    builder.set_surface_tension(0.072, "N/m")
    builder.set_density(1000, "kg/m^3")
    builder.set_phase_index([0, 1])
    strategy = builder.build()
    np.testing.assert_array_equal(strategy.phase_index, np.array([0, 1]))


def test_optional_tables_not_required():
    """Surface builders accept surface tension and temperature tables."""
    builder = SurfaceStrategyMolarBuilder()
    builder.set_surface_tension(0.072, "N/m")
    builder.set_density(1000, "kg/m^3")
    builder.set_molar_mass(0.01815, "kg/mol")
    builder.set_surface_tension_table(np.array([0.07, 0.08]), "N/m")
    builder.set_temperature_table(np.array([298.0, 308.0]), "K")
    strategy = builder.build()
    assert strategy.get_name() == "SurfaceStrategyMolar"
    np.testing.assert_allclose(builder.surface_tension_table, [0.07, 0.08])
    np.testing.assert_allclose(builder.temperature_table, [298.0, 308.0])
