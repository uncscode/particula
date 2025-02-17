"""Tests for the wall loss rate dynamics module."""

import numpy as np
from particula.dynamics.wall_loss import (
    spherical_wall_loss_rate,
    rectangle_wall_loss_rate,
)


def test_spherical_wall_loss_rate():
    """Test the spherical wall loss rate function."""
    # Test case 1
    wall_eddy_diffusivity = 0.1
    particle_radius = 1e-6
    particle_density = 1000.0
    particle_concentration = 100.0 * 1e6
    temperature = 298.0
    pressure = 101325.0
    chamber_radius = 10.0

    result = spherical_wall_loss_rate(
        wall_eddy_diffusivity,
        particle_radius,
        particle_density,
        particle_concentration,
        temperature,
        pressure,
        chamber_radius,
    )

    expected_loss_rate = -964.512508874514
    assert np.isclose(result, expected_loss_rate)

    # Test case 2
    wall_eddy_diffusivity = 0.05
    particle_radius = np.array([1e-9, 2e-6, 3e-3])
    particle_density = np.array([1000.0, 2000.0, 3000.0])
    particle_concentration = np.array([100.0, 200.0, 300.0]) * 1e6
    temperature = 273.15
    pressure = 100000.0
    chamber_radius = 5.0

    result = spherical_wall_loss_rate(
        wall_eddy_diffusivity,
        particle_radius,
        particle_density,
        particle_concentration,
        temperature,
        pressure,
        chamber_radius,
    )
    expected_loss_rate = np.array(
        [-9.22955735e03, -3.16327419e04, -1.54304331e11]
    )
    assert np.allclose(result, expected_loss_rate)


def test_rectangle_wall_loss_rate():
    """Test the rectangle wall loss rate function."""
    # Test single float input
    results = rectangle_wall_loss_rate(
        wall_eddy_diffusivity=0.1,
        particle_radius=1e-6,
        particle_density=1000.0,
        particle_concentration=100.0 * 1e6,
        temperature=298.0,
        pressure=101325.0,
        chamber_dimensions=(10.0, 10.0, 10.0),
    )

    expected_loss_rate = -1314.6750692789078
    assert np.isclose(results, expected_loss_rate)

    # Test single array input
    results = rectangle_wall_loss_rate(
        wall_eddy_diffusivity=0.05,
        particle_radius=np.array([1e-9, 2e-6, 3e-3]),
        particle_density=np.array([1000.0, 2000.0, 3000.0]),
        particle_concentration=np.array([100.0, 200.0, 300.0]) * 1e6,
        temperature=273.15,
        pressure=100000.0,
        chamber_dimensions=(5.0, 5.0, 5.0),
    )

    expected_loss_rate = np.array(
        [-1.84776108e04, -4.22330075e04, -2.05739107e11]
    )
    assert np.allclose(results, expected_loss_rate)
