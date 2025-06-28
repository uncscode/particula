"""Tests for the stokes_number module."""

import numpy as np
import pytest

from particula.particles.properties.stokes_number import get_stokes_number


def test_get_stokes_number_scalar():
    """Test get_stokes_number with scalar inputs."""
    particle_inertia_time = 0.02  # s
    kolmogorov_time = 0.005  # s

    expected = particle_inertia_time / kolmogorov_time
    result = get_stokes_number(particle_inertia_time, kolmogorov_time)

    assert np.isclose(result, expected, atol=1e-10)


def test_get_stokes_number_array():
    """Test get_stokes_number with NumPy array inputs."""
    particle_inertia_time = np.array([0.02, 0.03])
    kolmogorov_time = np.array([0.005, 0.004])

    expected = particle_inertia_time / kolmogorov_time
    result = get_stokes_number(particle_inertia_time, kolmogorov_time)

    assert np.allclose(result, expected, atol=1e-10)


def test_get_stokes_number_invalid():
    """Test that get_stokes_number raises errors for invalid inputs."""
    with pytest.raises(ValueError):
        get_stokes_number(-0.02, 0.005)  # Negative inertia time

    with pytest.raises(ValueError):
        get_stokes_number(0.02, -0.005)  # Negative Kolmogorov time

    with pytest.raises(ValueError):
        get_stokes_number(0.02, 0)  # Zero Kolmogorov time (division by zero)
