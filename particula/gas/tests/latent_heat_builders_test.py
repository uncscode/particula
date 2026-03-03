"""Tests for latent heat builder classes."""

import logging

import numpy as np
import pytest
from particula.abc_builder import BuilderABC
from particula.gas.latent_heat_builders import (
    ConstantLatentHeatBuilder,
    LinearLatentHeatBuilder,
    PowerLawLatentHeatBuilder,
)
from particula.gas.latent_heat_strategies import (
    ConstantLatentHeat,
    LinearLatentHeat,
    PowerLawLatentHeat,
)


def test_constant_builder_build() -> None:
    """Constant builder builds a ConstantLatentHeat strategy."""
    builder = ConstantLatentHeatBuilder().set_latent_heat_ref(2.26e6, None)
    strategy = builder.build()
    assert isinstance(strategy, ConstantLatentHeat)
    assert strategy.latent_heat(300.0) == pytest.approx(2.26e6)


def test_constant_builder_missing_param() -> None:
    """Missing latent_heat_ref raises ValueError when building."""
    with pytest.raises(ValueError, match="latent_heat_ref"):
        ConstantLatentHeatBuilder().build()


def test_constant_builder_negative_latent_heat_raises() -> None:
    """Negative latent heat is rejected by validate_inputs."""
    builder = ConstantLatentHeatBuilder()
    with pytest.raises(ValueError):
        builder.set_latent_heat_ref(-1.0, None)


def test_linear_builder_build() -> None:
    """Linear builder returns expected latent heat values."""
    builder = (
        LinearLatentHeatBuilder()
        .set_latent_heat_ref(2.501e6, None)
        .set_slope(2.3e3, None)
        .set_temperature_ref(273.15, None)
    )
    strategy = builder.build()
    assert isinstance(strategy, LinearLatentHeat)
    expected = 2.501e6 - 2.3e3 * (293.15 - 273.15)
    assert strategy.latent_heat(293.15) == pytest.approx(expected)


def test_linear_builder_missing_param() -> None:
    """Missing temperature_ref raises ValueError when building."""
    builder = (
        LinearLatentHeatBuilder()
        .set_latent_heat_ref(2.501e6, None)
        .set_slope(2.3e3, None)
    )
    with pytest.raises(ValueError, match="temperature_ref"):
        builder.build()


def test_linear_builder_slope_nonfinite_raises() -> None:
    """Non-finite slopes are rejected."""
    builder = LinearLatentHeatBuilder()
    with pytest.raises(ValueError):
        builder.set_slope(np.nan, None)


def test_linear_builder_temperature_negative_raises() -> None:
    """Negative reference temperature is rejected."""
    builder = LinearLatentHeatBuilder()
    with pytest.raises(ValueError):
        builder.set_temperature_ref(-5.0, None)


@pytest.mark.parametrize(
    "builder_class, parameters",
    [
        (
            ConstantLatentHeatBuilder,
            {
                "latent_heat_ref": 2.26e6,
                "latent_heat_ref_units": "J/kg",
            },
        ),
        (
            LinearLatentHeatBuilder,
            {
                "latent_heat_ref": 2.501e6,
                "latent_heat_ref_units": "J/kg",
                "slope": 2.3e3,
                "slope_units": "J/(kg*K)",
                "temperature_ref": 273.15,
                "temperature_ref_units": "K",
            },
        ),
        (
            PowerLawLatentHeatBuilder,
            {
                "latent_heat_ref": 2.257e6,
                "latent_heat_ref_units": "J/kg",
                "critical_temperature": 647.1,
                "critical_temperature_units": "K",
                "beta": 0.38,
            },
        ),
    ],
)
def test_builders_set_parameters_dict(
    builder_class: type[BuilderABC],
    parameters: dict[str, float | str],
) -> None:
    """Builders accept dict parameters with optional unit keys."""
    builder = builder_class()
    builder.set_parameters(parameters).build()


def test_linear_builder_set_parameters_missing_key_raises() -> None:
    """Missing required key in set_parameters triggers ValueError."""
    builder = LinearLatentHeatBuilder()
    parameters = {
        "latent_heat_ref": 2.501e6,
        "slope": 2.3e3,
    }
    with pytest.raises(ValueError, match="temperature_ref"):
        builder.set_parameters(parameters)


def test_power_law_builder_build() -> None:
    """Power law builder returns expected latent heat values."""
    builder = (
        PowerLawLatentHeatBuilder()
        .set_latent_heat_ref(2.257e6, None)
        .set_critical_temperature(647.1, None)
        .set_beta(0.38, None)
    )
    strategy = builder.build()
    assert isinstance(strategy, PowerLawLatentHeat)
    expected = 2.257e6 * (1 - 373.15 / 647.1) ** 0.38
    assert strategy.latent_heat(373.15) == pytest.approx(expected)


def test_power_law_builder_missing_param() -> None:
    """Missing beta raises ValueError when building."""
    builder = (
        PowerLawLatentHeatBuilder()
        .set_latent_heat_ref(2.257e6, None)
        .set_critical_temperature(647.1, None)
    )
    with pytest.raises(ValueError, match="beta"):
        builder.build()


def test_power_law_builder_negative_beta_raises() -> None:
    """Negative beta is rejected by validate_inputs."""
    builder = PowerLawLatentHeatBuilder()
    with pytest.raises(ValueError):
        builder.set_beta(-0.1)


def test_power_law_builder_round_trip() -> None:
    """Latent heat clamps to zero above critical temperature."""
    strategy = (
        PowerLawLatentHeatBuilder()
        .set_latent_heat_ref(2.257e6, None)
        .set_critical_temperature(647.1, None)
        .set_beta(0.38, None)
        .build()
    )
    assert strategy.latent_heat(700.0) == 0.0


def test_unit_conversion_latent_heat_ref() -> None:
    """Latent heat conversion applies when units are provided."""
    pytest.importorskip("pint")
    builder = ConstantLatentHeatBuilder().set_latent_heat_ref(2.26, "kJ/kg")
    assert builder.latent_heat_ref == pytest.approx(2.26e3)


def test_unit_conversion_temperature_ref() -> None:
    """Temperature conversion applies when units are provided."""
    pytest.importorskip("pint")
    builder = LinearLatentHeatBuilder().set_temperature_ref(1.0, "degC")
    assert builder.temperature_ref == pytest.approx(274.15)


def test_unit_conversion_temperature_ref_zero_offset_units() -> None:
    """Zero degC converts to a positive Kelvin temperature."""
    pytest.importorskip("pint")
    builder = LinearLatentHeatBuilder().set_temperature_ref(0.0, "degC")
    assert builder.temperature_ref == pytest.approx(273.15)


def test_unit_conversion_invalid_units_raises() -> None:
    """Invalid units raise pint UndefinedUnitError when pint is available."""
    pint = pytest.importorskip("pint")
    builder = ConstantLatentHeatBuilder()
    with pytest.raises(pint.errors.UndefinedUnitError):
        builder.set_latent_heat_ref(1.0, "not-a-unit")


def test_unit_conversion_linear_builder_values() -> None:
    """Linear builder converts slope and latent heat units when provided."""
    pytest.importorskip("pint")
    builder = (
        LinearLatentHeatBuilder()
        .set_latent_heat_ref(2.0, "kJ/kg")
        .set_slope(1.5, "kJ/(kg*K)")
        .set_temperature_ref(1.0, "degC")
    )
    assert builder.latent_heat_ref == pytest.approx(2000.0)
    assert builder.slope == pytest.approx(1500.0)
    assert builder.temperature_ref == pytest.approx(274.15)


def test_unit_conversion_power_law_builder_values() -> None:
    """Power-law builder converts latent heat and temperature units."""
    pytest.importorskip("pint")
    builder = (
        PowerLawLatentHeatBuilder()
        .set_latent_heat_ref(2.5, "kJ/kg")
        .set_critical_temperature(100.0, "degC")
        .set_beta(0.5)
    )
    assert builder.latent_heat_ref == pytest.approx(2500.0)
    assert builder.critical_temperature == pytest.approx(373.15)


def test_unit_conversion_power_law_builder_zero_offset_temperature() -> None:
    """Zero degC converts before validation of critical temperature."""
    pytest.importorskip("pint")
    builder = PowerLawLatentHeatBuilder().set_critical_temperature(0.0, "degC")
    assert builder.critical_temperature == pytest.approx(273.15)


def test_power_law_beta_units_warning_emitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Power-law builder logs a warning when beta units are provided."""
    builder = PowerLawLatentHeatBuilder()
    logger = logging.getLogger("particula")
    original_propagate = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING):
            builder.set_beta(0.5, "dimensionless")
    finally:
        logger.propagate = original_propagate
    assert "Ignoring units for beta" in caplog.text
