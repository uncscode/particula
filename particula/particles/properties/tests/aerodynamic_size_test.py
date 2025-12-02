"""Tests for the aerodynamic size module."""

import numpy as np
import pytest

from particula.particles.properties.aerodynamic_size import (
    get_aerodynamic_length,
    get_aerodynamic_shape_factor,
)


def test_particle_aerodynamic_length_single_value():
    """Verify that the particle_aerodynamic_length function calculates the
    correct aerodynamic length for a single particle.
    """
    physical_length = 0.00005  # 50 micrometers
    physical_slip_correction_factor = 1.1
    aerodynamic_slip_correction_factor = 1.0
    density = 1000  # kg/m^3
    expected_length = physical_length * np.sqrt(
        (physical_slip_correction_factor / aerodynamic_slip_correction_factor)
        * (density / 1000)
    )
    actual_length = get_aerodynamic_length(
        physical_length,
        physical_slip_correction_factor,
        aerodynamic_slip_correction_factor,
        density,
    )
    assert np.isclose(actual_length, expected_length), (
        "The value does not match."
    )


def test_particle_aerodynamic_length_array_input():
    """Test that the particle_aerodynamic_length function handles numpy array
    inputs correctly.
    """
    physical_length = np.array([0.00005, 0.00007])  # Array of lengths
    physical_slip_correction_factor = np.array([1.1, 1.2])
    aerodynamic_slip_correction_factor = np.array([1.0, 1.1])
    density = np.array([1000, 1200])
    expected_length = physical_length * np.sqrt(
        (physical_slip_correction_factor / aerodynamic_slip_correction_factor)
        * (density / 1000)
    )
    actual_length = get_aerodynamic_length(
        physical_length,
        physical_slip_correction_factor,
        aerodynamic_slip_correction_factor,
        density,
    )
    assert np.allclose(actual_length, expected_length)


@pytest.mark.parametrize(
    "shape_key, expected_factor",
    [
        ("sphere", 1.0),
        ("cube", 1.08),
        ("cylinder_avg_aspect_2", 1.1),
        ("cylinder_avg_aspect_5", 1.35),
        ("cylinder_avg_aspect_10", 1.68),
        ("spheres_cluster_3", 1.15),
        ("spheres_cluster_4", 1.17),
        ("bituminous_coal", 1.08),
        ("quartz", 1.36),
        ("sand", 1.57),
        ("talc", 1.88),
    ],
)
def test_get_aerodynamic_shape_factor(shape_key, expected_factor):
    """Test the get_aerodynamic_shape_factor function."""
    actual_factor = get_aerodynamic_shape_factor(shape_key)
    assert actual_factor == expected_factor, "The shape factor does not match."


def test_get_aerodynamic_shape_factor_invalid():
    """Test that the get_aerodynamic_shape_factor function raises a ValueError
    for unknown shapes.
    """
    with pytest.raises(ValueError):
        get_aerodynamic_shape_factor("unknown_shape")
