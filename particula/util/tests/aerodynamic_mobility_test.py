"""Test aerodynamic mobility functions."""

import numpy as np
import pytest
from particula.util.aerodynamic_mobility import particle_aerodynamic_mobility


def test_basic_mobility():
    """Test the basic mobility calculation."""
    radius = 0.5  # meters
    scf = 2.0  # dimensionless
    dv = 1.0  # Pascal-second
    expected = scf / (6 * np.pi * dv * radius)
    assert np.isclose(
        particle_aerodynamic_mobility(radius, scf, dv), expected)


def test_mobility_with_ndarray():
    """Test the mobility calculation with NumPy arrays."""
    radius = np.array([0.5, 1.0, 2.0])
    scf = np.array([2.0, 2.0, 2.0])
    dv = 1.0  # Pascal-second
    expected = scf / (6 * np.pi * dv * radius)
    np.testing.assert_allclose(
        particle_aerodynamic_mobility(
            radius, scf, dv), expected)


def test_mobility_with_zero_radius():
    """Test the mobility calculation with a zero radius."""
    radius = 0
    scf = 2.0
    dv = 1.0
    with pytest.raises(ZeroDivisionError):
        particle_aerodynamic_mobility(radius, scf, dv)


def test_mobility_with_negative_values():
    """Test the mobility calculation with negative values."""
    radius = -0.5
    scf = 2.0
    dv = 1.0
    expected = scf / (6 * np.pi * dv * radius)
    assert np.isclose(
        particle_aerodynamic_mobility(radius, scf, dv), expected)


def test_mobility_type_error():
    """Test the mobility calculation with incorrect types."""
    radius = '0.5'  # Incorrect type (string)
    scf = 2.0
    dv = 1.0
    with pytest.raises(TypeError):
        particle_aerodynamic_mobility(radius, scf, dv)
