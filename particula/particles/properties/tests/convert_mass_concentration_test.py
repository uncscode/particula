"""Tests for mass_concentration.py module."""

import numpy as np
import pytest
from particula.particles.properties import (
    convert_mass_concentration,
)


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
            np.array([0.5, 0.5]),  # 1D input with equal masses and molar masses
        ),
        (
            np.array([[100, 200], [50, 150]]),  # 2D input
            np.array([10, 20]),  # 1D molar masses, broadcast across rows
            np.array([[0.5, 0.5], [0.4, 0.6]]),
        ),  # Expected 2D mole fractions
    ],
)
def test_mass_concentration_to_mole_fraction(
    mass_concentrations, molar_masses, expected
):
    """Test mass_concentration_to_mole_fraction function."""
    mole_fractions = convert_mass_concentration.get_mole_fraction_from_mass(
        mass_concentrations, molar_masses
    )
    np.testing.assert_allclose(mole_fractions, expected, rtol=1e-5)


def test_mass_concentration_to_mole_fraction_zero_total_returns_zeros():
    """Zero total moles return zeros for 1D and zero rows in 2D inputs."""
    one_dimensional = convert_mass_concentration.get_mole_fraction_from_mass(
        np.array([0.0, 0.0]),
        np.array([10.0, 20.0]),
    )
    two_dimensional = convert_mass_concentration.get_mole_fraction_from_mass(
        np.array([[0.0, 0.0], [10.0, 30.0]]),
        np.array([10.0, 30.0]),
    )

    np.testing.assert_array_equal(one_dimensional, np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        two_dimensional,
        np.array([[0.0, 0.0], [0.5, 0.5]]),
        rtol=1e-5,
    )


def test_mass_concentration_to_mole_fraction_rejects_invalid_dimensions():
    """Three-dimensional mass concentrations are rejected explicitly."""
    with pytest.raises(
        ValueError, match="mass_concentrations must be either 1D or 2D"
    ):
        convert_mass_concentration.get_mole_fraction_from_mass(
            np.ones((1, 1, 1)),
            np.ones((1, 1, 1)),
        )


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
    mass_concentrations, densities, expected
):
    """Test mass_concentration_to_volume_fraction function."""
    volume_fractions = convert_mass_concentration.get_volume_fraction_from_mass(
        mass_concentrations, densities
    )
    np.testing.assert_allclose(volume_fractions, expected, rtol=1e-5)


def test_mass_concentration_to_volume_fraction_zero_total_returns_zeros():
    """Zero total volume returns zeros for 1D and zero rows in 2D inputs."""
    one_dimensional = convert_mass_concentration.get_volume_fraction_from_mass(
        np.array([0.0, 0.0]),
        np.array([10.0, 20.0]),
    )
    two_dimensional = convert_mass_concentration.get_volume_fraction_from_mass(
        np.array([[0.0, 0.0], [10.0, 30.0]]),
        np.array([10.0, 30.0]),
    )

    np.testing.assert_array_equal(one_dimensional, np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        two_dimensional,
        np.array([[0.0, 0.0], [0.5, 0.5]]),
        rtol=1e-5,
    )


def test_mass_concentration_to_volume_fraction_rejects_invalid_dimensions():
    """Three-dimensional mass concentrations are rejected explicitly."""
    with pytest.raises(
        ValueError, match="mass_concentrations must be either 1D or 2D"
    ):
        convert_mass_concentration.get_volume_fraction_from_mass(
            np.ones((1, 1, 1)),
            np.ones((1, 1, 1)),
        )


@pytest.mark.parametrize(
    "mass_concentrations, molar_masses",
    [
        (np.array([100, 0]), np.array([10, 0])),  # Test zero molar mass
        (
            np.array([100, -100]),
            np.array([10, 20]),
        ),  # Negative mass concentration
        (np.array([100, 200]), np.array([np.nan, 20])),  # Negative molar mass
    ],
)
def test_error_handling_mass_to_mole(mass_concentrations, molar_masses):
    """Test error handling for mass_concentration_to_mole_fraction function."""
    with pytest.raises(Exception):  # noqa: B017
        convert_mass_concentration.get_mole_fraction_from_mass(
            mass_concentrations, molar_masses
        )


@pytest.mark.parametrize(
    "mass_concentrations, densities",
    [
        (np.array([100, 200]), np.array([0, 20])),  # Zero density
        (
            np.array([100, -200]),
            np.array([10, 20]),
        ),  # Negative mass concentration
        (np.array([100, 200]), np.array([10, -20])),  # Negative density
    ],
)
def test_error_handling_mass_to_volume(mass_concentrations, densities):
    """Test error handling for mass_concentration_to_volume_fraction."""
    with pytest.raises(Exception):  # noqa: B017
        convert_mass_concentration.get_volume_fraction_from_mass(
            mass_concentrations, densities
        )


@pytest.mark.parametrize(
    "mass_concentrations, expected",
    [
        (np.array([10.0, 30.0, 60.0]), np.array([0.1, 0.3, 0.6])),
        (
            np.array([[10.0, 30.0], [0.0, 0.0]]),
            np.array([[0.25, 0.75], [0.0, 0.0]]),
        ),
    ],
)
def test_mass_concentration_to_mass_fraction(mass_concentrations, expected):
    """Mass fractions are normalized across 1D and 2D inputs."""
    mass_fractions = convert_mass_concentration.get_mass_fraction_from_mass(
        mass_concentrations
    )

    np.testing.assert_allclose(mass_fractions, expected, rtol=1e-5)


def test_mass_concentration_to_mass_fraction_rejects_invalid_dimensions():
    """Three-dimensional mass concentrations are rejected explicitly."""
    with pytest.raises(
        ValueError, match="mass_concentrations must be either 1D or 2D"
    ):
        convert_mass_concentration.get_mass_fraction_from_mass(
            np.ones((1, 1, 1))
        )


def test_mass_concentration_to_mass_fraction_rejects_negative_values():
    """Negative mass concentrations fail input validation."""
    with pytest.raises(Exception):  # noqa: B017
        convert_mass_concentration.get_mass_fraction_from_mass(
            np.array([1.0, -1.0])
        )
