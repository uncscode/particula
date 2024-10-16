"""Test builders for activity strategies
for error and validation handling.

Correctness is tested by activity_strategies_test.py.
"""

import pytest
import numpy as np
from particula.particles.activity_builders import (
    ActivityIdealMolarBuilder,
    ActivityIdealMassBuilder,
    ActivityKappaParameterBuilder,
)


def test_build_ideal_activity_mass():
    """Test building an IdealActivityMass object."""
    builder = ActivityIdealMassBuilder()
    activity = builder.build()
    assert activity.get_name() == "ActivityIdealMass"


def test_build_ideal_activity_molar_parameter():
    """Test that providing a negative molar mass raises a ValueError."""
    builder = ActivityIdealMolarBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_molar_mass(-1)
    assert "Molar mass must be a positive value." in str(excinfo.value)

    # test positive molar mass
    builder.set_molar_mass(1)
    assert builder.molar_mass == 1

    # test array of molar masses
    builder.set_molar_mass(np.array([1, 2, 3]))
    np.testing.assert_array_equal(builder.molar_mass, np.array([1, 2, 3]))

    # test setting molar mass units
    builder.set_molar_mass(1, molar_mass_units="g/mol")
    assert builder.molar_mass == 1e-3

    # test setting molar mass units for array
    builder.set_molar_mass(np.array([1, 2, 3]), molar_mass_units="g/mol")
    np.testing.assert_array_equal(
        builder.molar_mass, np.array([1e-3, 2e-3, 3e-3])
    )


def test_build_ideal_activity_molar_dict():
    """Test building an IdealActivityMolar object."""
    builder_dict = ActivityIdealMolarBuilder()
    parameters = {"molar_mass": 1, "molar_mass_units": "kg/mol"}
    builder_dict.set_parameters(parameters)
    assert builder_dict.molar_mass == 1

    # build the object
    activity = builder_dict.build()
    assert activity.__class__.__name__ == "ActivityIdealMolar"


def test_build_ideal_activity_molar_missing_parameters():
    """Test building an IdealActivityMolar object with missing parameters."""
    builder_missing = ActivityIdealMolarBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder_missing.build()
    assert "Required parameter(s) not set: molar_mass" in str(excinfo.value)


def test_build_kappa_parameter_activity_set_kappa():
    """Testing setting kappa parameter."""
    builder = ActivityKappaParameterBuilder()
    builder.set_kappa(1)
    assert builder.kappa == 1

    # test setting kappa units
    builder.set_kappa(1)
    assert builder.kappa == 1

    # test setting kappa units for array
    builder.set_kappa(np.array([1, 2, 3]))
    np.testing.assert_array_equal(builder.kappa, np.array([1, 2, 3]))

    # test setting kappa units, have no effect
    builder.set_kappa(1, kappa_units="no_units")
    assert builder.kappa == 1

    # test negative kappa
    with pytest.raises(ValueError) as excinfo:
        builder.set_kappa(-1)
    assert "Kappa parameter must be a positive value." in str(excinfo.value)


def test_build_kappa_parameter_activity_set_density():
    """Testing setting density parameter."""
    builder = ActivityKappaParameterBuilder()
    builder.set_density(1)
    assert builder.density == pytest.approx(1, rel=1e-5)

    # test setting density units
    builder.set_density(1, density_units="g/cm^3")
    assert builder.density == pytest.approx(1e3, rel=1e-5)

    # test setting density units for array
    builder.set_density(np.array([1, 2, 3]), density_units="g/cm^3")
    np.testing.assert_allclose(
        builder.density, np.array([1e3, 2e3, 3e3]), atol=1e-5
    )

    # test negative density
    with pytest.raises(ValueError) as excinfo:
        builder.set_density(-1)
    assert "Density must be a positive value." in str(excinfo.value)


def test_build_kappa_parameter_activity_set_molar_mass():
    """Testing setting molar mass parameter."""
    builder = ActivityKappaParameterBuilder()
    builder.set_molar_mass(1)
    assert builder.molar_mass == 1

    # test setting molar mass units
    builder.set_molar_mass(1, molar_mass_units="g/mol")
    assert builder.molar_mass == pytest.approx(1e-3, rel=1e-5)

    # test setting molar mass units for array
    builder.set_molar_mass(np.array([1, 2, 3]), molar_mass_units="g/mol")
    np.testing.assert_allclose(
        builder.molar_mass, np.array([1e-3, 2e-3, 3e-3]), atol=1e-5
    )

    # test negative molar mass
    with pytest.raises(ValueError) as excinfo:
        builder.set_molar_mass(-1)
    assert "Molar mass must be a positive value." in str(excinfo.value)


def test_build_kappa_parameter_activity_set_water_index():
    """Testing setting water index"""
    builder = ActivityKappaParameterBuilder()
    builder.set_water_index(1)
    assert builder.water_index == 1

    # test ignore units
    builder.set_water_index(1, water_index_units="no_units")
    assert builder.water_index == 1


def test_build_kappa_parameter_activity_dict():
    """Test building a KappaParameterActivity object."""
    builder_dict = ActivityKappaParameterBuilder()
    parameters = {
        "kappa": np.array([1, 2, 3]),
        "density": np.array([1, 2, 3]),
        "molar_mass": np.array([1, 2, 3]),
        "water_index": 1,
    }
    builder_dict.set_parameters(parameters)
    np.testing.assert_allclose(
        builder_dict.kappa, parameters["kappa"], atol=1e-5
    )
    np.testing.assert_allclose(
        builder_dict.density, parameters["density"], atol=1e-5
    )
    np.testing.assert_allclose(
        builder_dict.molar_mass, parameters["molar_mass"], atol=1e-5
    )
    assert builder_dict.water_index == 1

    # build the object
    activity = builder_dict.build()
    assert activity.__class__.__name__ == "ActivityKappaParameter"

    # test missing parameters
    builder_missing2 = ActivityKappaParameterBuilder()
    parameters = {
        "kappa": np.array([0, 1]),
        "density": np.array([1, 1]),
        "molar_mass": np.array([1, 2]),
    }
    with pytest.raises(ValueError) as excinfo:
        builder_missing2.set_parameters(parameters)
    assert "Missing required parameter(s): water_index" in str(excinfo.value)
