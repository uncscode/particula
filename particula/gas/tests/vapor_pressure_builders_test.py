"""Tests for the VaporPressureBuilder class."""

import pytest
from particula.gas.vapor_pressure_builders import (
    AntoineBuilder,
    ClausiusClapeyronBuilder,
    ConstantBuilder,
    WaterBuckBuilder,
)


def test_antoine_set_positive_a():
    """Test setting a positive value for the coefficient 'a'."""
    builder = AntoineBuilder()
    assert builder.set_a(10).a == 10


def test_antoine_set_negative_a():
    """Test setting a negative value for the coefficient 'a'."""
    builder = AntoineBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_a(-5)
    assert "Coefficient 'a' must be a positive value." in str(excinfo.value)


def test_antoine_set_parameters_with_missing_key():
    """Test setting parameters with a missing key in the dictionary."""
    builder = AntoineBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_parameters({"a": 10, "b": 2000})
    assert "Missing required parameter(s): c" in str(excinfo.value)


def test_antoine_build_without_all_coefficients():
    """Test building the strategy without all coefficients set."""
    builder = AntoineBuilder().set_a(10).set_b(2000)
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert "Required parameter(s) not set: c" in str(excinfo.value)


def test_clausius_set_latent_heat_positive():
    """Test setting a positive value for the latent heat."""
    builder = ClausiusClapeyronBuilder()
    builder.set_latent_heat(100, "J/kg")
    assert builder.latent_heat == 100, "Latent heat should be set correctly"


def test_clausius_set_latent_heat_negative():
    """Test setting a negative value for the latent heat."""
    builder = ClausiusClapeyronBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_latent_heat(-100, "J/kg")
    assert "Latent heat must be a positive numeric value." in str(
        excinfo.value
    )


def test_clausius_set_temperature_initial_positive():
    """Test setting a positive value for the initial temperature."""
    builder = ClausiusClapeyronBuilder()
    builder.set_temperature_initial(300, "K")
    assert builder.temperature_initial == 300


def test_clausius_set_temperature_initial_negative():
    """Test setting a negative value for the initial temperature."""
    builder = ClausiusClapeyronBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_temperature_initial(-5, "K")
    assert "Initial temperature must be a positive numeric value." in str(
        excinfo.value
    )


def test_clausius_set_pressure_initial_positive():
    """Test setting a positive value for the initial pressure."""
    builder = ClausiusClapeyronBuilder()
    builder.set_pressure_initial(101325, "Pa")
    assert builder.pressure_initial == 101325


def test_clausius_set_pressure_initial_negative():
    """Test setting a negative value for the initial pressure."""
    builder = ClausiusClapeyronBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_pressure_initial(-101325, "Pa")
    assert "Initial pressure must be a positive numeric value." in str(
        excinfo.value
    )


def test_clausius_set_parameters_complete():
    """Test setting all parameters at once, with different pressure units."""
    builder = ClausiusClapeyronBuilder()
    parameters = {
        "latent_heat": 2260,
        "latent_heat_units": "J/kg",
        "temperature_initial": 373,
        "temperature_initial_units": "K",
        "pressure_initial": 1,
        "pressure_initial_units": "atm",
    }
    builder.set_parameters(parameters)
    assert builder.latent_heat == 2260
    assert builder.temperature_initial == 373
    assert builder.pressure_initial == 101325


def test_clausius_build_success():
    """Test building the strategy successfully."""
    builder = (
        ClausiusClapeyronBuilder()
        .set_latent_heat(2260, "J/kg")
        .set_temperature_initial(373, "K")
        .set_pressure_initial(101325, "Pa")
    )
    builder.build()


def test_clausius_build_failure():
    """Test building the strategy without all parameters set."""
    builder = ClausiusClapeyronBuilder()
    builder.set_latent_heat(2260, "J/kg").set_temperature_initial(373, "K")
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert "Required parameter(s) not set: pressure_initial" in str(
        excinfo.value
    )


def test_constant_set_vapor_pressure_positive():
    """Test setting a positive value for the constant vapor pressure."""
    builder = ConstantBuilder()
    builder.set_vapor_pressure(101325, "Pa")
    assert builder.vapor_pressure == 101325


def test_constant_set_vapor_pressure_negative():
    """Test setting a negative value for the constant vapor pressure."""
    builder = ConstantBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.set_vapor_pressure(-101325, "Pa")
    assert "Vapor pressure must be a positive value." in str(excinfo.value)


def test_constant_set_parameters_complete():
    """Test setting all parameters at once. With different pressure units."""
    builder = ConstantBuilder()
    parameters = {"vapor_pressure": 1, "vapor_pressure_units": "atm"}
    builder.set_parameters(parameters)
    assert builder.vapor_pressure == 101325


def test_constant_build_success():
    """Test building the strategy successfully."""
    builder = ConstantBuilder()
    builder.set_vapor_pressure(101325, "Pa")
    builder.build()


def test_constant_build_failure():
    """Test building the strategy without setting the vapor pressure."""
    builder = ConstantBuilder()
    with pytest.raises(ValueError) as excinfo:
        builder.build()
    assert "Required parameter(s) not set: vapor_pressure" in str(
        excinfo.value
    )


def test_build_water_buck():
    """Test building the WaterBuck strategy."""
    builder = WaterBuckBuilder()
    builder.build()
