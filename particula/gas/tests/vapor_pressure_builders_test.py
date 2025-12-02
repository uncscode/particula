"""Tests for the VaporPressureBuilder class."""

import pytest

from particula.gas.vapor_pressure_builders import (
    AntoineVaporPressureBuilder,
    ClausiusClapeyronVaporPressureBuilder,
    ConstantVaporPressureBuilder,
    SaturationConcentrationVaporPressureBuilder,
    TableVaporPressureBuilder,
    WaterBuckVaporPressureBuilder,
)
from particula.gas.vapor_pressure_strategies import TableVaporPressureStrategy


def test_antoine_set_positive_a():
    """Test setting a positive value for the coefficient 'a'."""
    builder = AntoineVaporPressureBuilder()
    assert builder.set_a(10).a == 10


def test_antoine_set_negative_a():
    """Test setting a negative value for the coefficient 'a'."""
    builder = AntoineVaporPressureBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_a(-5)
    assert "Coefficient 'a' must be a positive value." in str(excinfo.value)


def test_antoine_set_parameters_with_missing_key():
    """Test setting parameters with a missing key in the dictionary."""
    builder = AntoineVaporPressureBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_parameters({"a": 10, "b": 2000})
    assert "Missing required parameter(s): c" in str(excinfo.value)


def test_antoine_build_without_all_coefficients():
    """Test building the strategy without all coefficients set."""
    builder = AntoineVaporPressureBuilder().set_a(10).set_b(2000)
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert "Required parameter(s) not set: c" in str(excinfo.value)


def test_clausius_set_latent_heat_positive():
    """Test setting a positive value for the latent heat."""
    builder = ClausiusClapeyronVaporPressureBuilder()
    builder.set_latent_heat(100, "J/mol")
    assert builder.latent_heat == 100, "Latent heat should be set correctly"


def test_clausius_set_latent_heat_negative():
    """Test setting a negative value for the latent heat."""
    builder = ClausiusClapeyronVaporPressureBuilder()
    with pytest.raises(ValueError):
        builder.set_latent_heat(-100, "J/mol")


def test_clausius_set_temperature_initial_positive():
    """Test setting a positive value for the initial temperature."""
    builder = ClausiusClapeyronVaporPressureBuilder()
    builder.set_temperature_initial(300, "K")
    assert builder.temperature_initial == 300


def test_clausius_set_temperature_initial_negative():
    """Test setting a negative value for the initial temperature."""
    builder = ClausiusClapeyronVaporPressureBuilder()
    with pytest.raises(ValueError):
        builder.set_temperature_initial(-5, "K")


def test_clausius_set_pressure_initial_positive():
    """Test setting a positive value for the initial pressure."""
    builder = ClausiusClapeyronVaporPressureBuilder()
    builder.set_pressure_initial(101325, "Pa")
    assert builder.pressure_initial == 101325


def test_clausius_set_pressure_initial_negative():
    """Test setting a negative value for the initial pressure."""
    builder = ClausiusClapeyronVaporPressureBuilder()
    with pytest.raises(ValueError):
        builder.set_pressure_initial(-101325, "Pa")


def test_clausius_set_parameters_complete():
    """Test setting all parameters at once, with different pressure units."""
    builder = ClausiusClapeyronVaporPressureBuilder()
    parameters = {
        "latent_heat": 2260,
        "latent_heat_units": "J/mol",
        "temperature_initial": 373,
        "temperature_initial_units": "K",
        "pressure_initial": 101325,
        "pressure_initial_units": "Pa",
    }
    builder.set_parameters(parameters)
    assert builder.latent_heat == 2260
    assert builder.temperature_initial == 373
    assert builder.pressure_initial == 101325


def test_clausius_build_success():
    """Test building the strategy successfully."""
    builder = (
        ClausiusClapeyronVaporPressureBuilder()
        .set_latent_heat(2260, "J/mol")
        .set_temperature_initial(373, "K")
        .set_pressure_initial(101325, "Pa")
    )
    builder.build()


def test_clausius_build_failure():
    """Test building the strategy without all parameters set."""
    builder = ClausiusClapeyronVaporPressureBuilder()
    builder.set_latent_heat(2260, "J/mol").set_temperature_initial(373, "K")
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert "Required parameter(s) not set: pressure_initial" in str(
        excinfo.value
    )


def test_constant_set_vapor_pressure_positive():
    """Test setting a positive value for the constant vapor pressure."""
    builder = ConstantVaporPressureBuilder()
    builder.set_vapor_pressure(101325, "Pa")
    assert builder.vapor_pressure == 101325


def test_constant_set_vapor_pressure_negative():
    """Test setting a negative value for the constant vapor pressure."""
    builder = ConstantVaporPressureBuilder()
    with pytest.raises(ValueError):
        builder.set_vapor_pressure(-101325, "Pa")


def test_constant_set_parameters_complete():
    """Test setting all parameters at once. With different pressure units."""
    builder = ConstantVaporPressureBuilder()
    parameters = {"vapor_pressure": 101325, "vapor_pressure_units": "Pa"}
    builder.set_parameters(parameters)
    assert builder.vapor_pressure == 101325


def test_constant_build_success():
    """Test building the strategy successfully."""
    builder = ConstantVaporPressureBuilder()
    builder.set_vapor_pressure(101325, "Pa")
    builder.build()


def test_constant_build_failure():
    """Test building the strategy without setting the vapor pressure."""
    builder = ConstantVaporPressureBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert "Required parameter(s) not set: vapor_pressure" in str(excinfo.value)


def test_build_water_buck():
    """Test building the WaterBuck strategy."""
    builder = WaterBuckVaporPressureBuilder()
    builder.build()


# ----------------------------------------------------------------------
# SaturationConcentrationVaporPressureBuilder tests
# ----------------------------------------------------------------------


def test_saturation_set_concentration_positive():
    """Test setting a positive saturation concentration (default units)."""
    builder = SaturationConcentrationVaporPressureBuilder()
    builder.set_saturation_concentration(1e-6, "kg/m^3")
    assert builder.saturation_concentration == pytest.approx(1e-6)


def test_saturation_build_success():
    """Test successful build when all parameters are provided."""
    builder = (
        SaturationConcentrationVaporPressureBuilder()
        .set_saturation_concentration(1e-6, "kg/m^3")
        .set_molar_mass(0.018, "kg/mol")
        .set_temperature(298.15, "K")
    )
    builder.build()  # Should not raise


def test_saturation_build_failure_missing_param():
    """Test build failure when temperature is missing."""
    builder = (
        SaturationConcentrationVaporPressureBuilder()
        .set_saturation_concentration(1e-6, "kg/m^3")
        .set_molar_mass(0.018, "kg/mol")
    )
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert "Required parameter(s) not set: temperature" in str(excinfo.value)


def test_table_builder_set_table_and_build():
    """Test building TableVaporPressureStrategy with tables."""
    builder = (
        TableVaporPressureBuilder()
        .set_vapor_pressure_table([1000.0, 2000.0], "Pa")
        .set_temperature_table([300.0, 350.0], "K")
    )
    strategy = builder.build()
    assert isinstance(strategy, TableVaporPressureStrategy)


def test_table_builder_missing_temperature_table():
    """Ensure error raised when temperature table not set."""
    builder = TableVaporPressureBuilder().set_vapor_pressure_table(
        [1000.0, 2000.0], "Pa"
    )
    with pytest.raises(ValueError):
        builder.build()
