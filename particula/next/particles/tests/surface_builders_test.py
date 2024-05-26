"""Test builders for surface strategies, for error and validation handling.

Correctness is tested by surface_strategies_test.py.
"""

import pytest
import numpy as np
from particula.next.particles.surface_builders import (
    SurfaceStrategyMolarBuilder, SurfaceStrategyMassBuilder,
    SurfaceStrategyVolumeBuilder
)


def test_build_surface_strategy_molar_parameters():
    """Test that providing a negative molar mass raises a ValueError."""
    builder = SurfaceStrategyMolarBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_molar_mass(-1)
    assert "Molar mass must be a positive value." in str(excinfo.value)

    # test positive molar mass
    builder.set_molar_mass(1)
    assert builder.molar_mass == 1

    # test array of molar masses
    builder.set_molar_mass(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder.molar_mass, np.array([1, 2, 3]))

    # test setting molar mass units
    builder.set_molar_mass(1, molar_mass_units='g/mol')
    assert builder.molar_mass == pytest.approx(1e-3, 1e-6)

    # test setting molar mass units for array
    builder.set_molar_mass(np.array([1, 2, 3]), molar_mass_units='g/mol')
    np.testing.assert_allclose(
        builder.molar_mass, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6)

    # density tests
    with pytest.raises(ValueError) as excinfo:
        builder.set_density(-1)
    assert "Density must be a positive value." in str(excinfo.value)

    # test positive density
    builder.set_density(1)
    assert builder.density == 1

    # test array of densities
    builder.set_density(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder.density, np.array([1, 2, 3]))

    # test setting density units
    builder.set_density(1, density_units='g/cm^3')
    assert builder.density == pytest.approx(1e3, 1e-6)

    # test setting density units for array
    builder.set_density(np.array([1, 2, 3]), density_units='g/cm^3')
    np.testing.assert_allclose(
        builder.density, np.array([1e3, 2e3, 3e3]), atol=1e-6)

    # surface tension tests
    builder = SurfaceStrategyMolarBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_surface_tension(-1)
    assert "Surface tension must be a positive value." in str(excinfo.value)

    # test positive surface tension
    builder.set_surface_tension(1)
    assert builder.surface_tension == 1

    # test array of surface tensions
    builder.set_surface_tension(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder.surface_tension, np.array([1, 2, 3]))

    # test setting surface tension units
    builder.set_surface_tension(1, surface_tension_units='mN/m')
    assert builder.surface_tension == pytest.approx(1e-3, 1e-6)

    # test setting surface tension units for array
    builder.set_surface_tension(
        np.array([1, 2, 3]), surface_tension_units='mN/m')
    np.testing.assert_allclose(
        builder.surface_tension, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6)


def test_build_surface_strategy_molar_dict():
    """Test building with a dictionary."""
    builder = SurfaceStrategyMolarBuilder()
    parameters = {
        "surface_tension": 0.072,
        "density": 1000,
        "molar_mass": 0.01815
    }
    builder.set_parameters(parameters)
    assert builder.surface_tension == 0.072
    assert builder.density == 1000
    assert builder.molar_mass == 0.01815

    # build the object
    strategy = builder.build()
    assert strategy.__class__.__name__ == "SurfaceStrategyMolar"


def test_build_surface_strategy_molar_missing_parameters():
    """Test building with missing parameters."""
    builder = SurfaceStrategyMolarBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert (
        "Required parameter(s) not set: surface_tension, density, molar_mass"
        in str(excinfo.value)
    )


def test_build_surface_strategy_mass_parameters():
    """Test parameters for SurfaceStrategyMassBuilder."""
    builder = SurfaceStrategyMassBuilder()
    # test surface tensions
    with pytest.raises(ValueError) as excinfo:
        builder.set_surface_tension(-1)
    assert "Surface tension must be a positive value." in str(excinfo.value)

    # test positive surface tension
    builder.set_surface_tension(1)
    assert builder.surface_tension == 1

    # test array of surface tensions
    builder.set_surface_tension(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder.surface_tension, np.array([1, 2, 3]))

    # test setting surface tension units
    builder.set_surface_tension(1, surface_tension_units='mN/m')
    assert builder.surface_tension == pytest.approx(1e-3, 1e-6)

    # test setting surface tension units for array
    builder.set_surface_tension(
        np.array([1, 2, 3]), surface_tension_units='mN/m')
    np.testing.assert_allclose(
        builder.surface_tension, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6)

    # test densities
    with pytest.raises(ValueError) as excinfo:
        builder.set_density(-1)
    assert "Density must be a positive value." in str(excinfo.value)

    # test positive density
    builder.set_density(1)
    assert builder.density == 1

    # test array of densities
    builder.set_density(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder.density, np.array([1, 2, 3]))

    # test setting density units
    builder.set_density(1, density_units='g/cm^3')
    assert builder.density == pytest.approx(1e3, 1e-6)

    # test setting density units for array
    builder.set_density(np.array([1, 2, 3]), density_units='g/cm^3')
    np.testing.assert_allclose(builder.density, np.array([1e3, 2e3, 3e3]),
                               atol=1e-6)


def test_build_surface_strategy_mass_dict():
    """Test building with a dictionary."""
    builder = SurfaceStrategyMassBuilder()
    parameters = {
        "surface_tension": 0.072,
        "density": 1000
    }
    builder.set_parameters(parameters)
    assert builder.surface_tension == 0.072
    assert builder.density == 1000

    # build the object
    strategy = builder.build()
    assert strategy.__class__.__name__ == "SurfaceStrategyMass"


def test_build_surface_strategy_mass_missing_parameters():
    """Test building with missing parameters."""
    builder = SurfaceStrategyMassBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert "Required parameter(s) not set: surface_tension, density" in str(
        excinfo.value
    )


def test_build_surface_strategy_volume_parameters():
    """Test volume SurfaceStrategyVolumeBuilder parameters."""
    # test surface tensions
    builder = SurfaceStrategyVolumeBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_surface_tension(-1)
    assert "Surface tension must be a positive value." in str(excinfo.value)

    # test positive surface tension
    builder.set_surface_tension(1)
    assert builder.surface_tension == 1

    # test array of surface tensions
    builder.set_surface_tension(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder.surface_tension, np.array([1, 2, 3]))

    # test setting surface tension units
    builder.set_surface_tension(1, surface_tension_units='mN/m')
    assert builder.surface_tension == pytest.approx(1e-3, 1e-6)

    # test setting surface tension units for array
    builder.set_surface_tension(
        np.array([1, 2, 3]), surface_tension_units='mN/m')
    np.testing.assert_allclose(
        builder.surface_tension, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6)

    # test densities
    with pytest.raises(ValueError) as excinfo:
        builder.set_density(-1)
    assert "Density must be a positive value." in str(excinfo.value)

    # test positive density
    builder.set_density(1)
    assert builder.density == 1

    # test array of densities
    builder.set_density(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder.density, np.array([1, 2, 3]))

    # test setting density units
    builder.set_density(1, density_units='g/cm^3')
    assert builder.density == pytest.approx(1e3, 1e-6)

    # test setting density units for array
    builder.set_density(np.array([1, 2, 3]), density_units='g/cm^3')
    np.testing.assert_allclose(
        builder.density, np.array([1e3, 2e3, 3e3]), atol=1e-6)


def test_build_surface_strategy_volume_dict():
    """Test building with a dictionary."""
    builder = SurfaceStrategyVolumeBuilder()
    parameters = {
        "surface_tension": 0.072,
        "density": 1000
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
