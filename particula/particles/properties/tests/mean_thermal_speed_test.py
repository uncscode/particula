"""Tests for the mean_thermal_speed function."""

import numpy as np
import pytest

from particula.particles.properties.mean_thermal_speed_module import (
    get_mean_thermal_speed,
)
from particula.util.constants import BOLTZMANN_CONSTANT


def test_mean_thermal_speed_single_value():
    """Test mean_thermal_speed for single float mass and temperature inputs."""
    mass = 4.188790204786391e-12  # Example mass in kg
    temperature = 300  # Temperature in Kelvin
    expected_speed = np.sqrt(
        (8 * BOLTZMANN_CONSTANT * temperature) / (np.pi * mass)
    )
    assert np.isclose(get_mean_thermal_speed(mass, temperature), expected_speed)


def test_mean_thermal_speed_array_input():
    """Test mean_thermal_speed with numpy array mass and temperature."""
    mass = np.array([2.18e-25, 2.18e-25])
    temperature = np.array([300, 350])
    expected_speed = np.sqrt(
        (8 * BOLTZMANN_CONSTANT * temperature) / (np.pi * mass)
    )
    assert np.allclose(
        get_mean_thermal_speed(mass, temperature), expected_speed, rtol=1e-6
    )


def test_mean_thermal_speed_input_validation():
    """Ensure that providing incorrect input types to mean_thermal_speed raises
    a TypeError.
    """
    with pytest.raises(TypeError):
        get_mean_thermal_speed("not a number", "also not a number")
