"""Tests for the Reynolds number calculation."""

import numpy as np
import pytest

from particula.particles.properties.reynolds_number import (
    get_particle_reynolds_number,
)


def test_get_particle_reynolds_number_scalar():
    """Test get_particle_reynolds_number with scalar inputs."""
    particle_radius = 50e-6  # 50 microns
    particle_velocity = 0.1  # m/s
    kinematic_viscosity = 1.5e-5  # m²/s

    expected = (2 * particle_radius * particle_velocity) / kinematic_viscosity
    result = get_particle_reynolds_number(
        particle_radius, particle_velocity, kinematic_viscosity
    )

    assert np.isclose(result, expected, atol=1e-10)


def test_get_particle_reynolds_number_array():
    """Test get_particle_reynolds_number with NumPy array inputs."""
    particle_radius = np.array([50e-6, 100e-6])
    particle_velocity = np.array([0.1, 0.2])
    kinematic_viscosity = np.array([1.5e-5, 1.5e-5])

    expected = (2 * particle_radius * particle_velocity) / kinematic_viscosity
    result = get_particle_reynolds_number(
        particle_radius, particle_velocity, kinematic_viscosity
    )

    assert np.allclose(result, expected, atol=1e-10)


def test_get_particle_reynolds_number_invalid():
    """Test get_particle_reynolds_number raises errors for invalid inputs."""
    with pytest.raises(ValueError):
        get_particle_reynolds_number(-50e-6, 0.1, 1.5e-5)  # Negative radius

    with pytest.raises(ValueError):
        get_particle_reynolds_number(50e-6, -0.1, 1.5e-5)  # Negative velocity

    with pytest.raises(ValueError):
        get_particle_reynolds_number(50e-6, 0.1, -1.5e-5)  # Negative viscosity


def test_get_particle_reynolds_number_regime_classification():
    """Test Reynolds number classification into flow regimes."""
    re_p_stokes = get_particle_reynolds_number(
        10e-6, 0.001, 1.5e-5
    )  # Should be < 1
    re_p_transitional = get_particle_reynolds_number(
        100e-6, 0.5, 1.5e-5
    )  # Should be 1 < Re_p < 1000
    re_p_turbulent = get_particle_reynolds_number(
        500e-6, 50.0, 1.5e-5
    )  # Should be > 1000

    assert re_p_stokes < 1
    assert 1 <= re_p_transitional < 1000
    assert re_p_turbulent > 1000
