"""Test the BuilderABC and BuilderDensityMixin classes."""

# pylint: disable=redefined-outer-name

import pytest
import numpy as np
from particula.next.abc_builder import (
    BuilderABC, BuilderDensityMixin, BuilderSurfaceTensionMixin,
    BuilderMolarMassMixin, BuilderConcentrationMixin,
    BuilderTemperatureMixin, BuilderPressureMixin,
    BuilderMassMixin, BuilderRadiusMixin,
    BuilderChargeMixin,
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


def test_temperature_mixin():
    """Test the BuilderTemperatureMixin class."""
    builder_mixin = BuilderTemperatureMixin()

    # test setting temperature
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_temperature(-1)
    assert "Temperature must be above zero Kelvin." in str(excinfo.value)

    # test positive temperature
    builder_mixin.set_temperature(1)
    assert builder_mixin.temperature == 1

    # test setting temperature units
    builder_mixin.set_temperature(1, temperature_units='degC')
    assert builder_mixin.temperature == pytest.approx(274.15, 1e-6)

    # negative degC
    builder_mixin.set_temperature(-10, temperature_units='degC')
    assert builder_mixin.temperature == pytest.approx(263.15, 1e-6)


def test_pressure_mixin():
    """Test the BuilderPressureMixin class."""
    builder_mixin = BuilderPressureMixin()

    # test setting pressure
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_pressure(-1)
    assert "Pressure must be a positive value." in str(excinfo.value)

    # test positive pressure
    builder_mixin.set_pressure(102000)
    assert builder_mixin.pressure == 102000

    # test setting pressure units
    builder_mixin.set_pressure(1, pressure_units='kPa')
    assert builder_mixin.pressure == pytest.approx(1e3, 1e-6)

    # test setting pressure units for array
    builder_mixin.set_pressure(np.array([1, 2, 3]), pressure_units='kPa')
    np.testing.assert_allclose(
        builder_mixin.pressure, np.array([1e3, 2e3, 3e3]), atol=1e-6)


def test_mass_mixin():
    """Test the BuilderMassMixin class."""
    builder_mixin = BuilderMassMixin()

    # test setting mass
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_mass(-1)
    assert "Mass must be a positive value." in str(excinfo.value)

    # test positive mass
    builder_mixin.set_mass(1)
    assert builder_mixin.mass == 1

    # test array of masses
    builder_mixin.set_mass(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder_mixin.mass, np.array([1, 2, 3]))

    # test setting mass units
    builder_mixin.set_mass(1, mass_units='g')
    assert builder_mixin.mass == pytest.approx(1e-3, 1e-6)

    # test setting mass units for array
    builder_mixin.set_mass(np.array([1, 2, 3]), mass_units='g')
    np.testing.assert_allclose(
        builder_mixin.mass, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6)


def test_radius_mixin():
    """Test the BuilderRadiusMixin class."""
    builder_mixin = BuilderRadiusMixin()

    # test setting radius
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_radius(-1)
    assert "Radius must be a positive value." in str(excinfo.value)

    # test positive radius
    builder_mixin.set_radius(1)
    assert builder_mixin.radius == 1

    # test array of radii
    builder_mixin.set_radius(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder_mixin.radius, np.array([1, 2, 3]))

    # test setting radius units
    builder_mixin.set_radius(1, radius_units='cm')
    assert builder_mixin.radius == pytest.approx(1e-2, 1e-6)

    # test setting radius units for array
    builder_mixin.set_radius(np.array([1, 2, 3]), radius_units='cm')
    np.testing.assert_allclose(
        builder_mixin.radius, np.array([1e-2, 2e-2, 3e-2]), atol=1e-6)


def test_charge_mixin():
    """Test the BuilderChargeMixin class."""
    builder_mixin = BuilderChargeMixin()

    # test setting charge
    builder_mixin.set_charge(1)
    assert builder_mixin.charge == 1

    # test array of charges
    builder_mixin.set_charge(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder_mixin.charge, np.array([1, 2, 3]))
