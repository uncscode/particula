"""Tests for latent heat strategy implementations."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas.latent_heat_strategies import (
    ConstantLatentHeat,
    LatentHeatStrategy,
    LinearLatentHeat,
    PowerLawLatentHeat,
)


def test_abc_cannot_instantiate():
    """Ensure the abstract strategy cannot be instantiated."""
    with pytest.raises(TypeError):
        LatentHeatStrategy()  # type: ignore[abstract]


def test_constant_scalar_return():
    """Return constant latent heat for scalar temperature."""
    strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
    result = strategy.latent_heat(300.0)
    assert result == pytest.approx(2.26e6)


def test_constant_array_broadcast():
    """Broadcast constant latent heat for array input."""
    strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
    temps = np.array([280.0, 290.0, 300.0])
    result = strategy.latent_heat(temps)
    npt.assert_array_almost_equal(result, np.full_like(temps, 2.26e6))


def test_constant_type_consistency():
    """Ensure scalar returns float and array returns NDArray."""
    strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
    scalar_result = strategy.latent_heat(298.15)
    array_result = strategy.latent_heat(np.array([298.15, 300.0]))
    assert isinstance(scalar_result, float)
    assert isinstance(array_result, np.ndarray)


def test_constant_zero_dimensional_array_returns_scalar():
    """Return float for 0-D NumPy array input."""
    strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
    result = strategy.latent_heat(np.array(298.15))
    assert isinstance(result, float)
    assert result == pytest.approx(2.26e6)


@pytest.mark.parametrize("latent_heat_ref", [2.26e6, 8.46e5])
def test_constant_different_values(latent_heat_ref):
    """Support different constant latent heat values."""
    strategy = ConstantLatentHeat(latent_heat_ref=latent_heat_ref)
    assert strategy.latent_heat(310.0) == pytest.approx(latent_heat_ref)


def test_constant_edge_cases_empty_and_zero_temperature():
    """Handle empty arrays and zero temperature input."""
    strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
    empty_result = strategy.latent_heat(np.array([]))
    npt.assert_array_equal(empty_result, np.array([], dtype=np.float64))
    assert strategy.latent_heat(0.0) == pytest.approx(2.26e6)


@pytest.mark.parametrize("latent_heat_ref", [0.0, -1.0, np.nan, np.inf])
def test_constant_latent_heat_ref_validation(latent_heat_ref):
    """Reject non-positive or non-finite latent heat references."""
    with pytest.raises(ValueError, match="latent_heat_ref"):
        ConstantLatentHeat(latent_heat_ref=latent_heat_ref)


def test_linear_scalar_return():
    """Return reference latent heat at the reference temperature."""
    strategy = LinearLatentHeat(
        latent_heat_ref=2.501e6,
        slope=2.3e3,
        temperature_ref=273.15,
    )
    result = strategy.latent_heat(273.15)
    assert result == pytest.approx(2.501e6)


def test_linear_known_water_values():
    """Match water reference values across common temperatures."""
    strategy = LinearLatentHeat(
        latent_heat_ref=2.501e6,
        slope=2.3e3,
        temperature_ref=273.15,
    )
    temperatures = np.array([273.15, 293.15, 313.15])
    expected_values = np.array([2.501e6, 2.455e6, 2.409e6])
    results = strategy.latent_heat(temperatures)
    npt.assert_allclose(results, expected_values, rtol=1e-3)


def test_linear_array_broadcast():
    """Broadcast the linear model for array input."""
    strategy = LinearLatentHeat(
        latent_heat_ref=2.501e6,
        slope=2.3e3,
        temperature_ref=273.15,
    )
    temperatures = np.array([273.15, 293.15, 313.15])
    expected = 2.501e6 - 2.3e3 * (temperatures - 273.15)
    results = np.asarray(strategy.latent_heat(temperatures))
    assert results.shape == temperatures.shape
    npt.assert_allclose(results, expected)


def test_linear_type_consistency():
    """Ensure scalar returns float and array returns NDArray."""
    strategy = LinearLatentHeat(
        latent_heat_ref=2.501e6,
        slope=2.3e3,
        temperature_ref=273.15,
    )
    scalar_result = strategy.latent_heat(300.0)
    array_result = strategy.latent_heat(np.array([300.0, 310.0]))
    assert isinstance(scalar_result, float)
    assert isinstance(array_result, np.ndarray)


def test_linear_zero_dimensional_array_returns_scalar():
    """Return float for 0-D NumPy array input."""
    strategy = LinearLatentHeat(
        latent_heat_ref=2.501e6,
        slope=2.3e3,
        temperature_ref=273.15,
    )
    result = strategy.latent_heat(np.array(300.0))
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "latent_heat_ref,slope,temperature_ref",
    [
        (0.0, 2.3e3, 273.15),
        (-1.0, 2.3e3, 273.15),
        (2.501e6, np.nan, 273.15),
        (2.501e6, np.inf, 273.15),
        (2.501e6, 2.3e3, 0.0),
        (2.501e6, 2.3e3, -10.0),
    ],
)
def test_linear_constructor_validation(latent_heat_ref, slope, temperature_ref):
    """Reject non-positive or non-finite linear strategy inputs."""
    with pytest.raises(ValueError):
        LinearLatentHeat(
            latent_heat_ref=latent_heat_ref,
            slope=slope,
            temperature_ref=temperature_ref,
        )


def test_power_law_scalar():
    """Compute a scalar latent heat using the power law model."""
    strategy = PowerLawLatentHeat(
        latent_heat_ref=2.257e6,
        critical_temperature=647.1,
        beta=0.38,
    )
    temperature = 373.15
    expected = 2.257e6 * (1 - temperature / 647.1) ** 0.38
    result = strategy.latent_heat(temperature)
    assert result == pytest.approx(expected)


def test_power_law_at_critical_temp():
    """Clamp latent heat to zero at the critical temperature."""
    strategy = PowerLawLatentHeat(
        latent_heat_ref=2.257e6,
        critical_temperature=647.1,
        beta=0.38,
    )
    assert strategy.latent_heat(647.1) == pytest.approx(0.0)


def test_power_law_above_critical_temp():
    """Clamp latent heat to zero above the critical temperature."""
    strategy = PowerLawLatentHeat(
        latent_heat_ref=2.257e6,
        critical_temperature=647.1,
        beta=0.38,
    )
    assert strategy.latent_heat(700.0) == pytest.approx(0.0)


def test_power_law_negative_temperature_clips():
    """Clamp negative temperatures to the reference latent heat."""
    strategy = PowerLawLatentHeat(
        latent_heat_ref=2.257e6,
        critical_temperature=647.1,
        beta=0.38,
    )
    assert strategy.latent_heat(-10.0) == pytest.approx(2.257e6)


def test_power_law_array_broadcast():
    """Broadcast the power law model for array input."""
    strategy = PowerLawLatentHeat(
        latent_heat_ref=2.257e6,
        critical_temperature=647.1,
        beta=0.38,
    )
    temperatures = np.array([273.15, 373.15, 647.1])
    ratio = np.clip(temperatures / 647.1, 0.0, 1.0)
    expected = 2.257e6 * (1 - ratio) ** 0.38
    results = np.asarray(strategy.latent_heat(temperatures))
    assert results.shape == temperatures.shape
    npt.assert_allclose(results, expected)


def test_power_law_type_consistency():
    """Ensure scalar returns float and array returns NDArray."""
    strategy = PowerLawLatentHeat(
        latent_heat_ref=2.257e6,
        critical_temperature=647.1,
        beta=0.38,
    )
    scalar_result = strategy.latent_heat(373.15)
    array_result = strategy.latent_heat(np.array([373.15, 400.0]))
    assert isinstance(scalar_result, float)
    assert isinstance(array_result, np.ndarray)


def test_power_law_zero_dimensional_array_returns_scalar():
    """Return float for 0-D NumPy array input."""
    strategy = PowerLawLatentHeat(
        latent_heat_ref=2.257e6,
        critical_temperature=647.1,
        beta=0.38,
    )
    result = strategy.latent_heat(np.array(373.15))
    assert isinstance(result, float)


def test_power_law_beta_zero_at_critical_temperature():
    """Return zero at the critical temperature when beta is zero."""
    strategy = PowerLawLatentHeat(
        latent_heat_ref=2.257e6,
        critical_temperature=647.1,
        beta=0.0,
    )
    assert strategy.latent_heat(647.1) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "latent_heat_ref,critical_temperature,beta",
    [
        (0.0, 647.1, 0.38),
        (-1.0, 647.1, 0.38),
        (2.257e6, 0.0, 0.38),
        (2.257e6, -10.0, 0.38),
        (2.257e6, 647.1, -0.1),
    ],
)
def test_power_law_constructor_validation(
    latent_heat_ref, critical_temperature, beta
):
    """Reject invalid power-law strategy inputs."""
    with pytest.raises(ValueError):
        PowerLawLatentHeat(
            latent_heat_ref=latent_heat_ref,
            critical_temperature=critical_temperature,
            beta=beta,
        )
