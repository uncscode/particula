"""Tests for mixins in coagulation_builder_mixin.py."""

import numpy as np
import pytest

# pylint: disable=line-too-long
from particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin import (  # noqa: E501
    BuilderDistributionTypeMixin,
    BuilderFluidDensityMixin,
    BuilderTurbulentDissipationMixin,
)


class MixinTester(
    BuilderDistributionTypeMixin,
    BuilderTurbulentDissipationMixin,
    BuilderFluidDensityMixin,
):
    """Simple class to inherit and test the mixins directly."""


def test_distribution_type_valid():
    """Test setting a valid distribution type."""
    tester = MixinTester()
    tester.set_distribution_type("discrete")
    assert tester.distribution_type == "discrete"


def test_distribution_type_invalid():
    """Test that an invalid distribution type raises ValueError."""
    tester = MixinTester()
    with pytest.raises(ValueError):
        tester.set_distribution_type("invalid_type")


def test_distribution_type_units():
    """Test that providing distribution_type_units logs a warning."""
    tester = MixinTester()
    tester.set_distribution_type(
        "discrete", distribution_type_units="dimensionless"
    )
    assert tester.distribution_type == "discrete"


def test_turbulent_dissipation_negative():
    """Test that negative turbulent dissipation raises ValueError."""
    tester = MixinTester()
    with pytest.raises(ValueError):
        tester.set_turbulent_dissipation(-1, "m^2/s^3")


def test_turbulent_dissipation_positive():
    """Test that a positive turbulent dissipation is set correctly."""
    tester = MixinTester()
    tester.set_turbulent_dissipation(1.0, "m^2/s^3")
    assert tester.turbulent_dissipation == pytest.approx(1.0)


def test_fluid_density_negative():
    """Test that negative fluid density raises ValueError."""
    tester = MixinTester()
    with pytest.raises(ValueError):
        tester.set_fluid_density(-1, "kg/m^3")


def test_fluid_density_positive():
    """Test that a valid fluid density is stored correctly."""
    tester = MixinTester()
    tester.set_fluid_density(1.2, "kg/m^3")
    assert tester.fluid_density == pytest.approx(1.2)


def test_fluid_density_array():
    """Test that an array of fluid densities is set correctly."""
    tester = MixinTester()
    density_array = np.array([1.2, 1.0, 0.9])
    tester.set_fluid_density(density_array, "kg/m^3")
    np.testing.assert_array_almost_equal(tester.fluid_density, density_array)  # type: ignore
