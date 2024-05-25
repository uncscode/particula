"""Tests for the AtmosphereBuilder class."""

import pytest
import numpy as np
from particula.next.gas.atmosphere_builder import AtmosphereBuilder
from particula.next.gas.species import GasSpecies
from particula.next.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy)


def test_gas_builder_with_species():
    """Test building a Gas object with the GasBuilder."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(
        vapor_pressure=np.array([101325, 101325]))
    names = np.array(["Oxygen", "Nitrogen"])
    molar_masses = np.array([0.032, 0.028])  # kg/mol
    condensables = np.array([False, False])
    concentrations = np.array([1.2, 0.8])  # kg/m^3

    gas_species = GasSpecies(
        name=names,
        molar_mass=molar_masses,
        vapor_pressure_strategy=vapor_pressure_strategy,
        condensable=condensables,
        concentration=concentrations
    )

    atmo = (
        AtmosphereBuilder()
        .set_temperature(298.15)
        .set_total_pressure(101325)
        .add_species(gas_species)
        .build()
    )

    assert atmo.temperature == 298.15
    assert atmo.total_pressure == 101325
    assert len(atmo.species) == 1


def test_gas_builder_without_species_raises_error():
    """Test that building a Gas object without any species raises an error."""
    builder = AtmosphereBuilder()
    # Omit adding any species to trigger the validation error
    with pytest.raises(ValueError) as e:
        builder.build()
    assert "Atmosphere must contain at least one species." in str(e.value)


def test_set_temperature():
    """Test setting the temperature of the atmosphere."""
    builder = AtmosphereBuilder()
    builder.set_temperature(300.0)
    assert builder.temperature == 300.0

    # test with unit conversion
    builder.set_temperature(300.0, temperature_units='C')
    assert builder.temperature == 573.15

    with pytest.raises(ValueError):
        builder.set_temperature(-10.0)


def test_set_total_pressure():
    """Test setting the total pressure of the atmosphere."""
    builder = AtmosphereBuilder()
    builder.set_total_pressure(102000.0)
    assert builder.total_pressure == 102000.0

    # test with unit conversion
    builder.set_total_pressure(102000.0, pressure_units='kPa')
    assert builder.total_pressure == 102000.0 * 1000.0

    with pytest.raises(ValueError):
        builder.set_total_pressure(-1000.0)
