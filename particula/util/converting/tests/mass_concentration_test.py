"""Tests for mass_concentration.py module"""

import numpy as np
import pytest

from particula.util.converting import convert_mass_concentration


@pytest.mark.parametrize(
    "mass_concentrations, molar_masses, expected",
    [
        (
            np.array([100, 200]),
            np.array([10, 20]),  # 1D input
            np.array([0.5, 0.5]),
        ),
        (
            np.array([50, 150, 200]),
            np.array([10, 30, 40]),  # 1D input
            np.array([0.333333, 0.333333, 0.333333]),
        ),
        (
            np.array([1, 1]),
            np.array([1, 1]),
            np.array(  # 1D input with equal masses and molar masses
                [0.5, 0.5]
            ),
        ),
        (
            np.array([[100, 200], [50, 150]]),  # 2D input
            np.array([10, 20]),  # 1D molar masses, broadcast across rows
            np.array([[0.5, 0.5], [0.4, 0.6]]),
        ),  # Expected 2D mole fractions
    ],
)
def test_mass_concentration_to_mole_fraction(
        mass_concentrations, molar_masses, expected):
    """Test mass_concentration_to_mole_fraction function"""
    mole_fractions = convert_mass_concentration.to_mole_fraction(
        mass_concentrations, molar_masses)
    np.testing.assert_allclose(mole_fractions, expected, rtol=1e-5)


@pytest.mark.parametrize(
    "mass_concentrations, densities, expected",
    [
        (
            np.array([100, 200]),  # 1D mass concentrations
            np.array([10, 20]),  # 1D densities
            np.array([0.5, 0.5]),
        ),  # expected volume fractions for 1D
        (np.array([50, 150]), np.array([5, 15]), np.array([0.5, 0.5])),
        (np.array([120, 180]), np.array([12, 18]), np.array([0.5, 0.5])),
        (
            np.array([[100, 200], [50, 150]]),  # 2D mass concentrations
            np.array([10, 20]),  # 1D densities (broadcasted over rows)
            np.array([[0.5, 0.5], [0.4, 0.6]]),
        ),  # expected volume fractions for 2D
    ],
)
def test_mass_concentration_to_volume_fraction(
        mass_concentrations, densities, expected):
    """Test mass_concentration_to_volume_fraction function"""
    volume_fractions = convert_mass_concentration.to_volume_fraction(
        mass_concentrations, densities)
    np.testing.assert_allclose(volume_fractions, expected, rtol=1e-5)


@pytest.mark.parametrize("mass_concentrations, molar_masses", [
    (np.array([100, 0]), np.array([10, 0])),  # Test zero molar mass
    (np.array([100, -100]), np.array([10, 20])),  # Negative mass concentration
    (np.array([100, 200]), np.array([-10, 20]))  # Negative molar mass
])
def test_error_handling_mass_to_mole(mass_concentrations, molar_masses):
    """Test error handling for mass_concentration_to_mole_fraction function"""
    with pytest.raises(Exception):
        convert_mass_concentration.to_mole_fraction(mass_concentrations, molar_masses)


@pytest.mark.parametrize("mass_concentrations, densities", [
    (np.array([100, 200]), np.array([0, 20])),  # Zero density
    (np.array([100, -200]), np.array([10, 20])),  # Negative mass concentration
    (np.array([100, 200]), np.array([10, -20]))  # Negative density
])
def test_error_handling_mass_to_volume(mass_concentrations, densities):
    """Test error handling for mass_concentration_to_volume_fraction
    function"""
    with pytest.raises(Exception):
        convert_mass_concentration.to_volume_fraction(mass_concentrations, densities)
