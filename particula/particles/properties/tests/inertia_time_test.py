"""Tests for the get_particle_inertia_time function."""

import numpy as np
import pytest

from particula.particles.properties.inertia_time import (
    get_particle_inertia_time,
)


def test_get_particle_inertia_time_scalar():
    """Test get_particle_inertia_time with scalar inputs."""
    particle_radius = 50e-6  # 50 microns
    particle_density = 1000  # kg/m³ (e.g., water, dust)
    fluid_density = 1.2  # kg/m³ (air)
    kinematic_viscosity = 1.5e-5  # m²/s

    expected = (
        (2 / 9)
        * (particle_density / fluid_density)
        * (particle_radius**2 / kinematic_viscosity)
    )

    result = get_particle_inertia_time(
        particle_radius,
        particle_density,
        fluid_density,
        kinematic_viscosity,
    )

    assert np.isclose(result, expected, atol=1e-10)


def test_get_particle_inertia_time_array():
    """Test get_particle_inertia_time with NumPy array inputs."""
    particle_radius = np.array([50e-6, 80e-6])
    particle_density = np.array([1000, 1200])
    fluid_density = np.array([1.2, 1.2])
    kinematic_viscosity = np.array([1.5e-5, 1.5e-5])

    expected = (
        (2 / 9)
        * (particle_density / fluid_density)
        * (particle_radius**2 / kinematic_viscosity)
    )

    result = get_particle_inertia_time(
        particle_radius,
        particle_density,
        fluid_density,
        kinematic_viscosity,
    )

    assert np.allclose(result, expected, atol=1e-10)


def test_get_particle_inertia_time_invalid():
    """Test that get_particle_inertia_time raises errors for invalid inputs."""
    with pytest.raises(ValueError):
        get_particle_inertia_time(-50e-6, 1000, 1.2, 1.5e-5)  # Negative radius

    with pytest.raises(ValueError):
        get_particle_inertia_time(
            50e-6, -1000, 1.2, 1.5e-5
        )  # Negative particle density

    with pytest.raises(ValueError):
        get_particle_inertia_time(
            50e-6, 1000, -1.2, 1.5e-5
        )  # Negative fluid density

    with pytest.raises(ValueError):
        get_particle_inertia_time(
            50e-6, 1000, 1.2, -1.5e-5
        )  # Negative viscosity
