"""Test the Kelvin effect module."""

import numpy as np
from particula.next.particles.properties.kelvin_effect_module import (
    kelvin_radius, kelvin_term
)


def test_prop_kelvin_radius_scalar():
    """Test kelvin_radius function with scalar inputs."""
    effective_surface_tension = 0.072  # N/m
    effective_density = 1000  # kg/m^3
    molar_mass = 0.018  # kg/mol
    temperature = 298  # K
    expected_radius = (2 * effective_surface_tension *
                       molar_mass) / (8.314 * temperature * effective_density)
    assert np.isclose(
        kelvin_radius(
            effective_surface_tension,
            effective_density,
            molar_mass,
            temperature),
        expected_radius)


def test_prop_kelvin_radius_array():
    """Test kelvin_radius function with array inputs."""
    effective_surface_tension = np.array([0.072, 0.072])
    effective_density = np.array([1000, 1500])
    molar_mass = np.array([0.018, 0.018])
    temperature = 298
    expected_radius = (2 * effective_surface_tension *
                       molar_mass) / (8.314 * temperature * effective_density)
    assert np.allclose(
        kelvin_radius(
            effective_surface_tension,
            effective_density,
            molar_mass,
            temperature),
        expected_radius)


def test_prop_kelvin_term_scalar():
    """Test kelvin_term function with scalar inputs."""
    radius = 0.5  # m
    kelvin_radius_value = 0.1  # m
    expected_term = np.exp(kelvin_radius_value / radius)
    assert np.isclose(kelvin_term(radius, kelvin_radius_value), expected_term)


def test_prop_kelvin_term_array():
    """Test kelvin_term function with array inputs."""
    radius = np.array([0.5, 1.0])
    kelvin_radius_value = np.array([0.1, 0.2])
    expected_term = np.exp(kelvin_radius_value / radius)
    assert np.allclose(kelvin_term(radius, kelvin_radius_value), expected_term)


def test_prop_negative_temperature_raises_error():
    """Ensure that negative temperature inputs raise an error."""
    with pytest.raises(ValueError):
        kelvin_radius(0.072, 1000, 0.018, -1)
