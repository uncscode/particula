"""Test the BuilderABC and BuilderDensityMixin classes."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from particula.builder_mixin import (
    BuilderChargeMixin,
    BuilderConcentrationMixin,
    BuilderDensityMixin,
    BuilderMassMixin,
    BuilderMolarMassMixin,
    BuilderPhaseIndexMixin,
    BuilderPressureMixin,
    BuilderRadiusMixin,
    BuilderSurfaceTensionMixin,
    BuilderSurfaceTensionTableMixin,
    BuilderTemperatureMixin,
    BuilderTemperatureTableMixin,
)


def test_density_mixin():
    """Test the BuilderDensityMixin class."""
    builder_mixin = BuilderDensityMixin()

    # test setting density
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_density(-1, "kg/m^3")
    assert "Argument 'density' must be positive." in str(excinfo.value)

    # test positive density
    builder_mixin.set_density(1, "kg/m^3")
    assert builder_mixin.density == 1

    # test array of densities
    builder_mixin.set_density(np.array([1, 2, 3]), "kg/m^3")
    np.testing.assert_allclose(builder_mixin.density, np.array([1, 2, 3]))

    # test setting density units
    builder_mixin.set_density(1e3, density_units="kg/m^3")
    assert builder_mixin.density == pytest.approx(1e3, 1e-6)


def test_surface_tension_mixin():
    """Test the BuilderSurfaceTensionMixin class."""
    builder_mixin = BuilderSurfaceTensionMixin()

    # test setting surface tension
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_surface_tension(-1, "N/m")
    assert "Argument 'surface_tension' must be positive." in str(excinfo.value)

    # test positive surface tension
    builder_mixin.set_surface_tension(1, "N/m")
    assert builder_mixin.surface_tension == 1

    # test array of surface tensions
    builder_mixin.set_surface_tension(np.array([1, 2, 3]), "N/m")
    result = builder_mixin.surface_tension
    np.testing.assert_allclose(result, np.array([1, 2, 3]))

    # test setting surface tension units
    builder_mixin.set_surface_tension(0.001, surface_tension_units="N/m")
    assert builder_mixin.surface_tension == pytest.approx(1e-3, 1e-6)

    # test setting surface tension units for array
    builder_mixin.set_surface_tension(
        np.array([1, 2, 3]) / 1000, surface_tension_units="N/m"
    )
    np.testing.assert_allclose(
        builder_mixin.surface_tension, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6
    )


def test_molar_mass_mixin():
    """Test the BuilderMolarMassMixin class."""
    builder_mixin = BuilderMolarMassMixin()

    # test positive molar mass
    builder_mixin.set_molar_mass(1, "kg/mol")
    assert builder_mixin.molar_mass == 1

    # test array of molar masses
    builder_mixin.set_molar_mass(np.array([1, 2, 3]), "kg/mol")
    np.testing.assert_allclose(builder_mixin.molar_mass, np.array([1, 2, 3]))

    # test setting molar mass units
    builder_mixin.set_molar_mass(1e-3, molar_mass_units="kg/mol")
    assert builder_mixin.molar_mass == pytest.approx(1e-3, 1e-6)

    # test setting molar mass units for array
    builder_mixin.set_molar_mass(np.array([1, 2, 3]) * 1e-3, "kg/mol")
    np.testing.assert_allclose(
        builder_mixin.molar_mass, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6
    )


def test_concentration_mixin():
    """Test the BuilderConcentrationMixin class."""
    builder_mixin = BuilderConcentrationMixin()

    # test setting concentration
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_concentration(-1, "kg/m^3")
    assert "Argument 'concentration' must be nonnegative." in str(excinfo.value)

    # test positive concentration
    builder_mixin.set_concentration(1, "kg/m^3")
    assert builder_mixin.concentration == 1

    # test array of concentrations
    builder_mixin.set_concentration(np.array([1, 2, 3]), "kg/m^3")
    np.testing.assert_allclose(builder_mixin.concentration, np.array([1, 2, 3]))

    # test setting concentration units
    builder_mixin.set_concentration(1 / 1000, concentration_units="kg/m^3")
    assert builder_mixin.concentration == pytest.approx(1e-3, 1e-6)

    # test setting concentration units for array
    builder_mixin.set_concentration(
        np.array([1, 2, 3]) / 1000, concentration_units="kg/m^3"
    )
    np.testing.assert_allclose(
        builder_mixin.concentration, np.array([1e-3, 2e-3, 3e-3]), atol=1e-6
    )


def test_temperature_mixin():
    """Test the BuilderTemperatureMixin class."""
    builder_mixin = BuilderTemperatureMixin()

    # test setting temperature in Kelvin
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_temperature(np.nan)
    assert "Argument 'temperature' must be finite (no inf or NaN)." in str(
        excinfo.value
    )

    # test positive temperature in Kelvin
    builder_mixin.set_temperature(300)
    assert builder_mixin.temperature == 300


def test_pressure_mixin():
    """Test the BuilderPressureMixin class."""
    builder_mixin = BuilderPressureMixin()

    # test setting pressure
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_pressure(-1, "Pa")
    assert "Argument 'pressure' must be nonnegative." in str(excinfo.value)

    # test positive pressure
    builder_mixin.set_pressure(102000, "Pa")
    assert builder_mixin.pressure == 102000

    # test setting pressure units
    builder_mixin.set_pressure(1e3, pressure_units="Pa")
    assert builder_mixin.pressure == pytest.approx(1e3, 1e-6)

    # test setting pressure units for array
    builder_mixin.set_pressure(np.array([1, 2, 3]) * 1e3, pressure_units="Pa")
    np.testing.assert_allclose(
        builder_mixin.pressure, np.array([1e3, 2e3, 3e3]), atol=1e-6
    )


def test_mass_mixin():
    """Test the BuilderMassMixin class."""
    builder_mixin = BuilderMassMixin()

    # test setting mass
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_mass(-1, "kg")
    assert "Argument 'mass' must be nonnegative." in str(excinfo.value)

    # test positive mass
    builder_mixin.set_mass(1, "kg")
    assert builder_mixin.mass == 1

    # test array of masses
    builder_mixin.set_mass(np.array([1, 2, 3]), "kg")
    np.testing.assert_allclose(builder_mixin.mass, np.array([1, 2, 3]))


def test_radius_mixin():
    """Test the BuilderRadiusMixin class."""
    builder_mixin = BuilderRadiusMixin()

    # test setting radius
    with pytest.raises(ValueError) as excinfo:
        builder_mixin.set_radius(-1, "m")
    assert "Argument 'radius' must be nonnegative." in str(excinfo.value)

    # test positive radius
    builder_mixin.set_radius(1, "m")
    assert builder_mixin.radius == 1

    # test array of radii
    builder_mixin.set_radius(np.array([1, 2, 3]), "m")
    np.testing.assert_allclose(builder_mixin.radius, np.array([1, 2, 3]))


def test_charge_mixin():
    """Test the BuilderChargeMixin class."""
    builder_mixin = BuilderChargeMixin()

    # test setting charge
    builder_mixin.set_charge(1)
    assert builder_mixin.charge == 1

    # test array of charges
    builder_mixin.set_charge(np.array([1, 2, 3]))
    np.testing.assert_allclose(builder_mixin.charge, np.array([1, 2, 3]))


def test_phase_index_mixin():
    """Test the BuilderPhaseIndexMixin class."""
    builder_mixin = BuilderPhaseIndexMixin()
    phase_index = np.array([0, 1, 0])
    builder_mixin.set_phase_index(phase_index)
    np.testing.assert_array_equal(builder_mixin.phase_index, phase_index)


def test_surface_tension_table_mixin():
    """Test the BuilderSurfaceTensionTableMixin class."""
    builder_mixin = BuilderSurfaceTensionTableMixin()

    with pytest.raises(ValueError):
        builder_mixin.set_surface_tension_table(np.array([-1.0, 2.0]))

    table = np.array([70.0, 80.0])
    builder_mixin.set_surface_tension_table(table * 1e-3, "N/m")
    np.testing.assert_allclose(
        builder_mixin.surface_tension_table, np.array([0.07, 0.08])
    )


def test_temperature_table_mixin():
    """Test the BuilderTemperatureTableMixin class."""
    builder_mixin = BuilderTemperatureTableMixin()

    builder_mixin.set_temperature_table(np.array([25.0, 35.0]) + 273.15, "K")
    np.testing.assert_allclose(
        builder_mixin.temperature_table,
        np.array([298.15, 308.15]),
        rtol=1e-5,
    )
