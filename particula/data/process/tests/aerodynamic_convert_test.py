"""
Tests for the aerodynamic_convert module.
"""

import numpy as np

from particula.data.process.aerodynamic_convert import (
    convert_aerodynamic_to_physical_radius,
    convert_physical_to_aerodynamic_radius,
)


def test_convert_aerodynamic_to_physical_radius():
    """Test the conversion from aerodynamic to physical radius."""
    aerodynamic_radius = np.array([1.0, 2.0, 3.0]) * 1e-6  # m
    pressure = 101325.0
    temperature = 298.0
    particle_density = 2000.0
    aerodynamic_shape_factor = 1.3
    reference_density = 1000.0

    physical_radius = convert_aerodynamic_to_physical_radius(
        aerodynamic_radius,
        pressure,
        temperature,
        particle_density,
        aerodynamic_shape_factor,
        reference_density,
    )

    expected_physical_radius = np.array(
        [7.98497748e-07, 1.60454568e-06, 2.41071053e-06]
    )
    assert np.allclose(physical_radius, expected_physical_radius)


def test_convert_physical_to_aerodynamic_radius():
    """Test the conversion from physical to aerodynamic radius."""
    physical_radius = np.array([1.0, 2.0, 3.0]) * 1e-6  # m
    pressure = 101325.0
    temperature = 298.0
    particle_density = 2000.0
    aerodynamic_shape_factor = 1.5
    reference_density = 1000.0

    aerodynamic_radius = convert_physical_to_aerodynamic_radius(
        physical_radius,
        pressure,
        temperature,
        particle_density,
        aerodynamic_shape_factor,
        reference_density,
    )

    expected_aerodynamic_radius = np.array(
        [1.16091860e-06, 2.31573747e-06, 3.47047854e-06]
    )
    assert np.allclose(aerodynamic_radius, expected_aerodynamic_radius)
