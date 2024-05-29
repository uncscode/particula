"""Test the BuilderABC and BuilderDensityMixin classes."""

# pylint: disable=redefined-outer-name

import pytest
import numpy as np
from particula.next.abc_builder import (
    BuilderABC, BuilderDensityMixin, BuilderSurfaceTensionMixin,
    BuilderMolarMassMixin, BuilderConcentrationMixin
)


# Example of a concrete class extending BuilderABC for testing
class ConcreteBuilder(BuilderABC):
    """Concrete class extending BuilderABC for testing purposes."""
    def build(self):
        return "Build successful!"


# Setup the fixture for the testing
@pytest.fixture
def builder_fix():
    """Fixture for the BuilderABC class."""
    return ConcreteBuilder(required_parameters=['param1', 'param2'])


def test_builder_init(builder_fix):
    """Test the initialization of the BuilderABC class."""
    assert builder_fix.required_parameters == [
        'param1', 'param2'], "Initialization of required parameters failed"


def test_check_keys_valid(builder_fix):
    """Check if the keys you want to set are present in the parameters"""
    parameters = {'param1': 10, 'param2': 20, 'param1_units': 'meters'}
    try:
        builder_fix.check_keys(parameters)
    except ValueError as e:
        pytest.fail(f"Unexpected ValueError raised: {e}")


def test_check_keys_missing_required(builder_fix):
    """Check if the keys you want to set are present in the parameters"""
    parameters = {'param1': 10}  # missing 'param2'
    with pytest.raises(ValueError) as excinfo:
        builder_fix.check_keys(parameters)
    assert 'Missing required parameter(s): param2' in str(excinfo.value)


def test_check_keys_invalid(builder_fix):
    """Check if the keys are invalid in the parameters dictionary."""
    parameters = {'param1': 10, 'param2': 20, 'param3': 30}
    with pytest.raises(ValueError) as excinfo:
        builder_fix.check_keys(parameters)
    assert "Trying to set an invalid parameter(s)" in str(excinfo.value)


def test_pre_build_check_complete(builder_fix):
    """Check if all required attribute parameters are set before building."""
    builder_fix.param1 = 10
    builder_fix.param2 = 20
    try:
        builder_fix.pre_build_check()
    except ValueError as e:
        pytest.fail(f"Unexpected ValueError raised: {e}")


def test_density_mixin():
    """Test the BuilderDensityMixin class."""
    builder_mixin = BuilderDensityMixin()

    # test setting density
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_density(-1)
    assert "Density must be a positive value." in str(excinfo.value)

    # test positive density
    builder_mixin.set_density(1)
    assert builder_mixin.density == 1

    # test array of densities
    builder_mixin.set_density(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder_mixin.density, np.array([1, 2, 3]))

    # test setting density units
    builder_mixin.set_density(1, density_units='g/cm^3')
    assert builder_mixin.density == pytest.approx(1e3, 1e-6)

    # test setting density units for array
    builder_mixin.set_density(np.array([1, 2, 3]), density_units='g/cm^3')
    np.testing.assert_allclose(
        builder_mixin.density, np.array([1e3, 2e3, 3e3]), atol=1e-6)


def test_surface_tension_mixin():
    """Test the BuilderSurfaceTensionMixin class."""
    builder_mixin = BuilderSurfaceTensionMixin()

    # test setting surface tension
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_surface_tension(-1)
    assert "Surface tension must be a positive value." in str(excinfo.value)

    # test positive surface tension
    builder_mixin.set_surface_tension(1)
    assert builder_mixin.surface_tension == 1

    # test array of surface tensions
    builder_mixin.set_surface_tension(np.array([1, 2, 3]))
    np.testing.assert_allclose(
        builder_mixin.surface_tension, np.array([1, 2, 3]))

    # test setting surface tension units
    builder_mixin.set_surface_tension(1, surface_tension_units='mN/m')
    assert builder_mixin.surface_tension == pytest.approx(1e-3, 1e-6)

    # test setting surface tension units for array
    builder_mixin.set_surface_tension(
        np.array([1, 2, 3]), surface_tension_units='mN/m')
    np.testing.assert_allclose(
        builder_mixin.surface_tension, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6)


def test_molar_mass_mixin():
    """Test the BuilderMolarMassMixin class."""
    builder_mixin = BuilderMolarMassMixin()

    # test positive molar mass
    builder_mixin.set_molar_mass(1)
    assert builder_mixin.molar_mass == 1

    # test array of molar masses
    builder_mixin.set_molar_mass(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder_mixin.molar_mass, np.array([1, 2, 3]))

    # test setting molar mass units
    builder_mixin.set_molar_mass(1, molar_mass_units='g/mol')
    assert builder_mixin.molar_mass == pytest.approx(1e-3, 1e-6)

    # test setting molar mass units for array
    builder_mixin.set_molar_mass(
        np.array([1, 2, 3]), molar_mass_units='g/mol')
    np.testing.assert_allclose(
        builder_mixin.molar_mass, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6)


def test_concentration_mixin():
    """Test the BuilderConcentrationMixin class."""
    builder_mixin = BuilderConcentrationMixin()

    # test setting concentration
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_concentration(-1)
    assert "Concentration must be a positive value." in str(excinfo.value)

    # test positive concentration
    builder_mixin.set_concentration(1)
    assert builder_mixin.concentration == 1

    # test array of concentrations
    builder_mixin.set_concentration(np.array([1, 2, 3]))
    np.testing.assert_allclose(
        builder_mixin.concentration, np.array([1, 2, 3]))

    # test setting concentration units
    builder_mixin.set_concentration(1, concentration_units='g/m^3')
    assert builder_mixin.concentration == pytest.approx(1e-3, 1e-6)

    # test setting concentration units for array
    builder_mixin.set_concentration(
        np.array([1, 2, 3]), concentration_units='g/m^3')
    np.testing.assert_allclose(
        builder_mixin.concentration, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6)
