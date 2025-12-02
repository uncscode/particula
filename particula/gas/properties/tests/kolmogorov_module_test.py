"""Test Kolmogorov Properties."""

import numpy as np
import pytest

from particula.gas.properties.kolmogorov_module import (
    get_kolmogorov_length,
    get_kolmogorov_time,
    get_kolmogorov_velocity,
)


def test_kolmogorov_time_float():
    """Test kolmogorov_time with float inputs."""
    assert np.isclose(get_kolmogorov_time(1.0, 1.0), 1.0)
    assert np.isclose(get_kolmogorov_time(0.01, 0.01), 1.0)
    assert np.isclose(get_kolmogorov_time(0.1, 0.01), 3.1622776601683795)


def test_kolmogorov_time_array():
    """Test kolmogorov_time with numpy array inputs."""
    kinematic_viscosity = np.array([1.0, 0.01, 0.1])
    turbulent_dissipation = np.array([1.0, 0.01, 0.01])
    expected = np.array([1.0, 1.0, 3.1622776601683795])
    result = get_kolmogorov_time(kinematic_viscosity, turbulent_dissipation)
    assert np.allclose(result, expected)


def test_kolmogorov_time_input_range():
    """Test kolmogorov_time with invalid input ranges."""
    with pytest.raises(ValueError):
        get_kolmogorov_time(-1.0, 1.0)
    with pytest.raises(ValueError):
        get_kolmogorov_time(1.0, -1.0)
    with pytest.raises(ValueError):
        get_kolmogorov_time(-1.0, -1.0)


def test_kolmogorov_length_float():
    """Test kolmogorov_length with float inputs."""
    assert np.isclose(get_kolmogorov_length(1.0, 1.0), 1.0)
    assert np.isclose(get_kolmogorov_length(0.01, 0.01), 0.1)
    assert np.isclose(get_kolmogorov_length(0.1, 0.01), 0.5623413251903491)


def test_kolmogorov_length_array():
    """Test kolmogorov_length with numpy array inputs."""
    kinematic_viscosity = np.array([1.0, 0.01, 0.1])
    turbulent_dissipation = np.array([1.0, 0.01, 0.01])
    expected = np.array([1.0, 0.1, 0.5623413251903491])
    result = get_kolmogorov_length(kinematic_viscosity, turbulent_dissipation)
    assert np.allclose(result, expected)


def test_kolmogorov_velocity_float():
    """Test kolmogorov_velocity with float inputs."""
    assert np.isclose(get_kolmogorov_velocity(1.0, 1.0), 1.0)
    assert np.isclose(get_kolmogorov_velocity(0.01, 0.01), 0.1)
    assert np.isclose(get_kolmogorov_velocity(0.1, 0.01), 0.1778279410038923)


def test_kolmogorov_velocity_array():
    """Test kolmogorov_velocity with numpy array inputs."""
    kinematic_viscosity = np.array([1.0, 0.01, 0.1])
    turbulent_dissipation = np.array([1.0, 0.01, 0.01])
    expected = np.array([1.0, 0.1, 0.1778279410038923])
    result = get_kolmogorov_velocity(kinematic_viscosity, turbulent_dissipation)
    assert np.allclose(result, expected)
