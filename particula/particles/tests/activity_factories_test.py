"""Test for activity factories. Is the factory function working as expected,
creating the right activity strategy based on the input parameters,
and raising the right exceptions when needed.

The Builder is tested independently.
"""

import numpy as np
import pytest

from particula.particles.activity_builders import (
    ActivityNonIdealBinaryBuilder,
)
from particula.particles.activity_factories import (
    ActivityFactory,
)
from particula.particles.activity_strategies import (
    ActivityIdealMass,
    ActivityIdealMolar,
    ActivityKappaParameter,
    ActivityNonIdealBinary,
)


def test_mass_ideal_strategy_no_parameters():
    """Test factory function for mass_ideal strategy without parameters."""
    strategy = ActivityFactory().get_strategy("mass_ideal")
    assert isinstance(strategy, ActivityIdealMass)


def test_molar_ideal_strategy_with_parameters():
    """Test factory function for molar_ideal strategy with parameters."""
    parameters = {
        "molar_mass": np.array([100.0, 200.0, 300.0]),
        "molar_mass_units": "kg/mol",
    }
    strategy = ActivityFactory().get_strategy("molar_ideal", parameters)
    assert isinstance(strategy, ActivityIdealMolar)
    np.testing.assert_allclose(
        strategy.molar_mass, parameters["molar_mass"], atol=1e-4
    )


def test_kappa_parameter_strategy_with_parameters():
    """Test factory function for kappa_parameter strategy with parameters."""
    parameters = {
        "kappa": np.array([0.1, 0.2, 0.3]),
        "density": np.array([1000.0, 2000.0, 3000.0]),
        "density_units": "kg/m^3",
        "molar_mass": np.array([1.0, 2.0, 3.0]),
        "molar_mass_units": "kg/mol",
        "water_index": 0,
    }
    strategy = ActivityFactory().get_strategy("kappa_parameter", parameters)
    assert isinstance(strategy, ActivityKappaParameter)
    np.testing.assert_allclose(strategy.kappa, parameters["kappa"], atol=1e-4)
    np.testing.assert_allclose(
        strategy.density, parameters["density"], atol=1e-4
    )
    np.testing.assert_allclose(
        strategy.molar_mass, parameters["molar_mass"], atol=1e-4
    )
    assert strategy.water_index == parameters["water_index"]


def test_invalid_strategy_type():
    """Test factory function with an invalid strategy type."""
    with pytest.raises(ValueError) as excinfo:
        ActivityFactory().get_strategy("invalid_type")
    assert "Unknown strategy type: invalid_type" in str(excinfo.value)


def _non_ideal_params(
    functional_group: str | None = None,
) -> dict[str, float | str]:
    params: dict[str, float | str] = {
        "molar_mass": 0.200,
        "oxygen2carbon": 0.5,
        "density": 1400.0,
    }
    if functional_group is not None:
        params["functional_group"] = functional_group
    return params


def test_activity_factory_non_ideal_binary_dispatch():
    """Test factory can dispatch to the non_ideal_binary strategy."""
    strategy = ActivityFactory().get_strategy(
        "non_ideal_binary", _non_ideal_params()
    )
    assert isinstance(strategy, ActivityNonIdealBinary)
    assert strategy.get_name() == "ActivityNonIdealBinary"


def test_activity_factory_non_ideal_binary_with_functional_group():
    """Test optional functional_group propagates to the strategy."""
    params = _non_ideal_params("alcohol")
    strategy = ActivityFactory().get_strategy("non_ideal_binary", params)
    assert strategy.functional_group == "alcohol"


def test_activity_factory_non_ideal_binary_missing_param_raises():
    """Test missing required parameters surfaces a ValueError."""
    params = {"molar_mass": 0.200, "density": 1400.0}
    with pytest.raises(ValueError):
        ActivityFactory().get_strategy("non_ideal_binary", params)


def test_activity_factory_get_builders_includes_non_ideal():
    """Ensure get_builders exposes the non_ideal_binary builder."""
    builders = ActivityFactory().get_builders()
    assert "non_ideal_binary" in builders
    assert isinstance(
        builders["non_ideal_binary"], ActivityNonIdealBinaryBuilder
    )


def test_activity_factory_non_ideal_binary_case_insensitive_key():
    """Test case-insensitive keys still dispatch correctly."""
    params = _non_ideal_params()
    strategy = ActivityFactory().get_strategy("Non_Ideal_Binary", params)
    assert isinstance(strategy, ActivityNonIdealBinary)


def test_activity_factory_kappa_alias_still_works():
    """Regression: alias 'kappa' should map to ActivityKappaParameter."""
    parameters = {
        "kappa": np.array([0.1, 0.0]),
        "density": np.array([1000.0, 1200.0]),
        "density_units": "kg/m^3",
        "molar_mass": np.array([0.018, 0.046]),
        "molar_mass_units": "kg/mol",
        "water_index": 0,
    }
    strategy = ActivityFactory().get_strategy("kappa", parameters)
    assert isinstance(strategy, ActivityKappaParameter)
