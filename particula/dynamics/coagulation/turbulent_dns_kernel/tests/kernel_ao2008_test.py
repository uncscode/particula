"""Tests that collision kernel function get_kernel_ao2008 can be evaluated."""

import unittest

import numpy as np

from particula.dynamics.coagulation.turbulent_dns_kernel.turbulent_dns_kernel_ao2008 import (  # noqa: E501
    get_turbulent_dns_kernel_ao2008,
)


class TestKernelAO2008(unittest.TestCase):
    # pylint: disable=too-many-instance-attributes
    """Unit tests for the collision kernel function get_kernel_ao2008."""

    def setUp(self):
        """Set up common test parameters."""
        self.stokes_number_scalar = 0.5  # [-]
        self.kolmogorov_length_scale = 1e-6  # [m]
        self.reynolds_lambda = 100  # [-]
        self.normalized_accel_variance = 11  # [-]
        self.kolmogorov_velocity = 0.01  # [m/s]
        self.kolmogorov_time = 0.005  # [s]

        # Array-based test inputs
        self.particle_radius_array = np.array([10e-6, 20e-6, 30e-6])  # [m]
        self.particle_inertia_time_array = np.array([0.02, 0.03, 0.05])  # [s]
        self.stokes_number_array = np.array([0.5, 0.6, 0.7])  # [-]
        self.velocity_dispersion_scalar = 0.1  # [m/s]

    def test_get_kernel_ao2008_array(self):
        """Test get_kernel_ao2008 with NumPy array inputs."""
        result = get_turbulent_dns_kernel_ao2008(
            self.particle_radius_array,
            self.velocity_dispersion_scalar,
            self.particle_inertia_time_array,
            self.stokes_number_array,
            self.kolmogorov_length_scale,
            self.reynolds_lambda,
            self.normalized_accel_variance,
            self.kolmogorov_velocity,
            self.kolmogorov_time,
        )

        self.assertEqual(
            result.shape,
            (3, 3),
            f"Expected shape (3, 3), but got {result.shape}",
        )

    def test_invalid_inputs(self):
        """Ensure validation errors are raised for invalid inputs."""
        with self.assertRaises(ValueError):
            get_turbulent_dns_kernel_ao2008(
                -1 * self.particle_radius_array,
                self.velocity_dispersion_scalar,
                self.particle_inertia_time_array,
                self.stokes_number_array,
                self.kolmogorov_length_scale,
                self.reynolds_lambda,
                self.normalized_accel_variance,
                self.kolmogorov_velocity,
                self.kolmogorov_time,
            )  # Negative radius

        with self.assertRaises(ValueError):
            get_turbulent_dns_kernel_ao2008(
                self.particle_radius_array,
                -1 * self.velocity_dispersion_scalar,
                self.particle_inertia_time_array,
                self.stokes_number_array,
                self.kolmogorov_length_scale,
                self.reynolds_lambda,
                self.normalized_accel_variance,
                self.kolmogorov_velocity,
                self.kolmogorov_time,
            )  # Negative velocity_dispersion

        with self.assertRaises(ValueError):
            get_turbulent_dns_kernel_ao2008(
                self.particle_radius_array,
                self.velocity_dispersion_scalar,
                -1 * self.particle_inertia_time_array,
                self.stokes_number_array,
                self.kolmogorov_length_scale,
                self.reynolds_lambda,
                self.normalized_accel_variance,
                self.kolmogorov_velocity,
                self.kolmogorov_time,
            )  # Negative particle_inertia_time

        with self.assertRaises(ValueError):
            get_turbulent_dns_kernel_ao2008(
                self.particle_radius_array,
                self.velocity_dispersion_scalar,
                self.particle_inertia_time_array,
                -1 * self.stokes_number_array,
                self.kolmogorov_length_scale,
                self.reynolds_lambda,
                self.normalized_accel_variance,
                self.kolmogorov_velocity,
                self.kolmogorov_time,
            )  # Negative stokes_number
