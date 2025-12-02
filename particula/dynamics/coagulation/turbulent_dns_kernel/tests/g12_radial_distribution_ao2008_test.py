# pylint: disable=duplicate-code
"""Test the radial distribution of the coagulation kernel g12 using the
AO2008 DNS model.

By calling the get_g12_radial_distribution_ao2008 function, we also
test the private functions used in the calculation of the radial
distribution function.
"""

import numpy as np
import pytest

from particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008 import (  # noqa: E501
    get_g12_radial_distribution_ao2008,
)


def test_get_g12_radial_distribution_ao2008_scalar():
    """Test get_g12_radial_distribution_ao2008 with a small array input."""
    particle_radius = np.array(
        [10e-6, 20e-6, 30e-6]
    )  # Particle radii in meters
    stokes_number = np.array([0.5, 0.7, 1.2])  # Stokes numbers
    kolmogorov_length_scale = 1e-6  # m
    reynolds_lambda = 100  # Taylor-microscale Reynolds number
    normalized_accel_variance = 11  # Normalized acceleration variance
    kolmogorov_velocity = 0.01  # m/s
    kolmogorov_time = 0.005  # s

    # Compute g12
    g12_matrix = get_g12_radial_distribution_ao2008(
        particle_radius,
        stokes_number,
        kolmogorov_length_scale,
        reynolds_lambda,
        normalized_accel_variance,
        kolmogorov_velocity,
        kolmogorov_time,
    )

    # Check the shape
    assert g12_matrix.shape == (
        3,
        3,
    ), f"Expected shape (3,3), but got {g12_matrix.shape}"

    # Ensure values are non-negative
    assert np.all(g12_matrix >= 0), "Expected all values to be non-negative"


def test_get_g12_radial_distribution_ao2008_invalid_inputs():
    """Test that get_g12_radial_distribution_ao2008 raises validation errors
    for invalid inputs.
    """
    particle_radius = np.array([10e-6, 20e-6, 30e-6])
    stokes_number = np.array([0.5, 0.7, 1.2])
    kolmogorov_length_scale = 1e-6
    reynolds_lambda = 100
    normalized_accel_variance = 11
    kolmogorov_velocity = 0.01
    kolmogorov_time = 0.005

    with pytest.raises(ValueError):
        get_g12_radial_distribution_ao2008(
            -particle_radius,  # Negative values should raise an error
            stokes_number,
            kolmogorov_length_scale,
            reynolds_lambda,
            normalized_accel_variance,
            kolmogorov_velocity,
            kolmogorov_time,
        )

    with pytest.raises(ValueError):
        get_g12_radial_distribution_ao2008(
            particle_radius,
            -stokes_number,  # should raise an error
            kolmogorov_length_scale,
            reynolds_lambda,
            normalized_accel_variance,
            kolmogorov_velocity,
            kolmogorov_time,
        )

    with pytest.raises(ValueError):
        get_g12_radial_distribution_ao2008(
            particle_radius,
            stokes_number,
            -kolmogorov_length_scale,  # should raise an error
            reynolds_lambda,
            normalized_accel_variance,
            kolmogorov_velocity,
            kolmogorov_time,
        )

    with pytest.raises(ValueError):
        get_g12_radial_distribution_ao2008(
            particle_radius,
            stokes_number,
            kolmogorov_length_scale,
            -reynolds_lambda,  # should raise an error
            normalized_accel_variance,
            kolmogorov_velocity,
            kolmogorov_time,
        )


def test_get_g12_radial_distribution_ao2008_edge_cases():
    """Test get_g12_radial_distribution_ao2008 with edge cases, such as
    very small or very large Stokes numbers.
    """
    particle_radius = np.array([10e-6, 20e-6, 30e-6])
    stokes_number = np.array(
        [0.000001, 0.01, 2000.0]
    )  # Includes extreme Stokes numbers
    kolmogorov_length_scale = 1e-6
    reynolds_lambda = 100
    normalized_accel_variance = 11
    kolmogorov_velocity = 0.01
    kolmogorov_time = 0.005

    g12_matrix = get_g12_radial_distribution_ao2008(
        particle_radius,
        stokes_number,
        kolmogorov_length_scale,
        reynolds_lambda,
        normalized_accel_variance,
        kolmogorov_velocity,
        kolmogorov_time,
    )

    assert g12_matrix.shape == (
        3,
        3,
    ), f"Expected shape (3,3), but got {g12_matrix.shape}"
    assert np.all(g12_matrix >= 0), "Expected all values to be non-negative"
