"""Tests for latent heat strategy implementations."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas.latent_heat_strategies import (
    ConstantLatentHeat,
    LatentHeatStrategy,
)


def test_abc_cannot_instantiate():
    """Ensure the abstract strategy cannot be instantiated."""
    with pytest.raises(TypeError):
        LatentHeatStrategy()


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
