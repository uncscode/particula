"""Tests for the mean_thermal_speed function."""
import pytest
import numpy as np
from particula.next.particles.properties import mean_thermal_speed
from particula.constants import BOLTZMANN_CONSTANT


def test_mean_thermal_speed_single_value():
    """
    Test mean_thermal_speed with single float inputs for mass and temperature.
    """
    mass = 2.18e-25  # Example mass in kg
    temperature = 300  # Temperature in Kelvin
    expected_speed = np.sqrt(
        (8 * BOLTZMANN_CONSTANT.m * temperature) / (np.pi * mass))
    assert np.isclose(mean_thermal_speed(mass, temperature),
                      expected_speed), "Calculated speed does not match expected value."


def test_mean_thermal_speed_array_input():
    """
    Test mean_thermal_speed with numpy array inputs for mass and temperature.
    """
    mass = np.array([2.18e-25, 2.18e-25])
    temperature = np.array([300, 350])
    expected_speed = np.sqrt(
        (8 * BOLTZMANN_CONSTANT.m * temperature) / (np.pi * mass))
    np.testing.assert_allclose(
        mean_thermal_speed(
            mass, temperature), expected_speed, rtol=1e-6), "Calculated speeds do not match expected values."


def test_mean_thermal_speed_input_validation():
    """
    Ensure that providing incorrect input types to mean_thermal_speed raises
    a TypeError.
    """
    with pytest.raises(TypeError):
        mean_thermal_speed("not a number", "also not a number")
