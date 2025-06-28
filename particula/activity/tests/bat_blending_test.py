"""Blending weights for the BAT model."""

import numpy as np
import pytest

from particula.activity.bat_blending import bat_blending_weights


def test_bat_blending_weights():
    """Test for bat_blending_weights function."""
    molar_mass_ratio = 18.016 / 250
    oxygen2carbon = 0.3

    weights = bat_blending_weights(
        molar_mass_ratio=molar_mass_ratio, oxygen2carbon=oxygen2carbon
    )
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)


def test_bat_blending_weights_float():
    """Test bat_blending_weights with float inputs."""
    molar_mass_ratio = 0.5
    oxygen2carbon = 1.0
    result = bat_blending_weights(molar_mass_ratio, oxygen2carbon)
    assert result.shape == (3,)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_bat_blending_weights_array():
    """Test bat_blending_weights with array inputs."""
    molar_mass_ratio = np.array([0.5, 0.6])
    oxygen2carbon = np.array([1.0, 1.2])
    result = bat_blending_weights(molar_mass_ratio, oxygen2carbon)
    assert result.shape == (2, 3)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_bat_blending_weights_edge_cases():
    """Test bat_blending_weights with edge cases."""
    molar_mass_ratio = 0.5
    oxygen2carbon = 0.0
    result = bat_blending_weights(molar_mass_ratio, oxygen2carbon)
    assert result.shape == (3,)
    assert np.all(result >= 0) and np.all(result <= 1)

    oxygen2carbon = 10.0
    result = bat_blending_weights(molar_mass_ratio, oxygen2carbon)
    assert result.shape == (3,)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_bat_blending_weights_invalid_inputs():
    """Test bat_blending_weights with invalid inputs."""
    molar_mass_ratio = -0.5
    oxygen2carbon = 1.0
    with pytest.raises(ValueError):
        bat_blending_weights(molar_mass_ratio, oxygen2carbon)

    molar_mass_ratio = 0.5
    oxygen2carbon = -1.0
    with pytest.raises(ValueError):
        bat_blending_weights(molar_mass_ratio, oxygen2carbon)
