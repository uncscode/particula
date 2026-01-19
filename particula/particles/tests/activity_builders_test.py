"""Test builders for activity strategies
for error and validation handling.

Correctness is tested by activity_strategies_test.py.
"""

import numpy as np
import pytest

from particula.particles.activity_builders import (
    ActivityIdealMassBuilder,
    ActivityIdealMolarBuilder,
    ActivityKappaParameterBuilder,
    ActivityNonIdealBinaryBuilder,
)
from particula.particles.activity_strategies import ActivityNonIdealBinary


def test_build_ideal_activity_mass():
    """Test building an IdealActivityMass object."""
    builder = ActivityIdealMassBuilder()
    activity = builder.build()
    assert activity.get_name() == "ActivityIdealMass"


def test_build_ideal_activity_molar_parameter():
    """Test that providing a negative molar mass raises a ValueError."""
    builder = ActivityIdealMolarBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_molar_mass(-1, "kg/mol")
    assert "Argument 'molar_mass' must be positive." in str(excinfo.value)

    # test positive molar mass
    builder.set_molar_mass(1, "kg/mol")
    assert builder.molar_mass == 1

    # test array of molar masses
    builder.set_molar_mass(np.array([1, 2, 3]), "kg/mol")
    np.testing.assert_array_equal(builder.molar_mass, np.array([1, 2, 3]))


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
    builder.set_density(1, "kg/m^3")
    assert builder.density == pytest.approx(1, rel=1e-5)

    # test setting density units
    builder.set_density(1e3, density_units="kg/m^3")
    assert builder.density == pytest.approx(1e3, rel=1e-5)

    # test setting density units for array
    builder.set_density(np.array([1, 2, 3]) * 1e3, density_units="kg/m^3")
    np.testing.assert_allclose(
        builder.density, np.array([1e3, 2e3, 3e3]), atol=1e-5
    )

    # test negative density
    with pytest.raises(ValueError) as excinfo:
        builder.set_density(-1, "kg/m^3")
    assert "Argument 'density' must be positive." in str(excinfo.value)


def test_build_kappa_parameter_activity_set_molar_mass():
    """Testing setting molar mass parameter."""
    builder = ActivityKappaParameterBuilder()
    builder.set_molar_mass(1, "kg/mol")
    assert builder.molar_mass == 1

    # test setting molar mass units
    builder.set_molar_mass(0.001, molar_mass_units="kg/mol")
    assert builder.molar_mass == pytest.approx(1e-3, rel=1e-5)

    # test setting molar mass units for array
    builder.set_molar_mass(
        np.array([1, 2, 3]) / 1000, molar_mass_units="kg/mol"
    )
    np.testing.assert_allclose(
        builder.molar_mass, np.array([1e-3, 2e-3, 3e-3]), atol=1e-5
    )

    # test negative molar mass
    with pytest.raises(ValueError):
        builder.set_molar_mass(-1, "kg/mol")


def test_build_kappa_parameter_activity_set_water_index():
    """Testing setting water index."""
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
        "density_units": "kg/m^3",
        "molar_mass": np.array([1, 2, 3]),
        "molar_mass_units": "kg/mol",
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


def test_activity_non_ideal_binary_builder_success_and_optional_group():
    """Build succeeds with required params and preserves optional group."""
    builder = ActivityNonIdealBinaryBuilder()
    strategy = (
        builder.set_molar_mass(0.200, "kg/mol")
        .set_oxygen2carbon(0.4)
        .set_density(1400.0, "kg/m^3")
        .set_functional_group("alcohol")
        .build()
    )
    assert isinstance(strategy, ActivityNonIdealBinary)
    assert strategy.functional_group == "alcohol"
    assert strategy.get_name() == "ActivityNonIdealBinary"


def test_activity_non_ideal_binary_builder_functional_group_list():
    """Optional functional_group accepts list and is preserved."""
    functional_groups = ["carboxylic_acid", "alcohol"]
    strategy = (
        ActivityNonIdealBinaryBuilder()
        .set_molar_mass(0.150, "kg/mol")
        .set_oxygen2carbon(0.5)
        .set_density(1300.0, "kg/m^3")
        .set_functional_group(functional_groups)
        .build()
    )
    assert strategy.functional_group == functional_groups


def test_activity_non_ideal_binary_builder_warns_on_units():
    """Units args for dimensionless params warn and are ignored."""
    builder = ActivityNonIdealBinaryBuilder()
    with pytest.warns(UserWarning):
        builder.set_oxygen2carbon(0.3, oxygen2carbon_units="ratio")
    with pytest.warns(UserWarning):
        builder.set_functional_group(
            "alcohol", functional_group_units="unitless"
        )


def test_activity_non_ideal_binary_builder_negative_oxygen2carbon_raises():
    """Negative oxygen2carbon raises ValueError."""
    builder = ActivityNonIdealBinaryBuilder()
    with pytest.raises(ValueError):
        builder.set_oxygen2carbon(-0.1)


def test_activity_non_ideal_binary_builder_missing_required_on_build():
    """Missing required parameters triggers pre_build_check error."""
    builder = ActivityNonIdealBinaryBuilder()
    builder.set_molar_mass(0.200, "kg/mol")
    builder.set_density(1400.0, "kg/m^3")
    with pytest.raises(ValueError):
        builder.build()


def test_activity_non_ideal_binary_builder_fluent_returns_self():
    """Fluent setters return the builder instance."""
    builder = ActivityNonIdealBinaryBuilder()
    assert builder.set_molar_mass(0.2, "kg/mol") is builder
    assert builder.set_density(1000.0, "kg/m^3") is builder
    assert builder.set_oxygen2carbon(0.3) is builder
    assert builder.set_functional_group("alcohol") is builder


def test_activity_non_ideal_binary_builder_unit_conversion_mixins():
    """Mixin setters perform unit conversion before build."""
    strategy = (
        ActivityNonIdealBinaryBuilder()
        .set_molar_mass(0.2, "kg/mol")
        .set_oxygen2carbon(0.4)
        .set_density(1400.0, "kg/m^3")
        .build()
    )
    assert strategy.molar_mass_kg == pytest.approx(0.2, rel=1e-6)
    assert strategy.density == pytest.approx(1400.0, rel=1e-6)


def test_activity_non_ideal_binary_builder_set_parameters_supports_units():
    """set_parameters handles units and optional functional_group."""
    builder = ActivityNonIdealBinaryBuilder()
    parameters = {
        "molar_mass": 0.2,
        "molar_mass_units": "kg/mol",
        "oxygen2carbon": 0.45,
        "density": 1250.0,
        "density_units": "kg/m^3",
        "functional_group": "ether",
    }
    builder.set_parameters(parameters)
    strategy = builder.build()
    assert strategy.functional_group == "ether"
    assert strategy.molar_mass_kg == pytest.approx(0.2, rel=1e-6)
    assert strategy.oxygen2carbon == pytest.approx(0.45, rel=1e-6)


def test_activity_non_ideal_binary_builder_set_parameters_missing_required():
    """set_parameters errors on missing required keys."""
    builder = ActivityNonIdealBinaryBuilder()
    with pytest.raises(ValueError):
        builder.set_parameters({"molar_mass": 0.2, "density": 1400.0})
