"""Test module for the Taylor microscale functions."""

import unittest

import numpy as np
import pytest

from particula.gas.properties.taylor_microscale_module import (
    get_lagrangian_taylor_microscale_time,
    get_taylor_microscale,
    get_taylor_microscale_reynolds_number,
)


def test_get_lagrangian_taylor_microscale_time_scalar():
    """Test get_lagrangian_taylor_microscale_time with scalar inputs."""
    kolmogorov_time = 0.01  # s
    re_lambda = 100  # Reynolds number
    ao = 11  # Normalized acceleration variance

    expected = kolmogorov_time * np.sqrt((2 * re_lambda) / (15**0.5 * ao))
    result = get_lagrangian_taylor_microscale_time(
        kolmogorov_time, re_lambda, ao
    )

    assert np.isclose(result, expected, atol=1e-10)


def test_get_lagrangian_taylor_microscale_time_array():
    """Test get_lagrangian_taylor_microscale_time with NumPy array inputs."""
    kolmogorov_time = np.array([0.01, 0.02])
    re_lambda = np.array([100, 200])
    ao = np.array([11, 12])

    expected = kolmogorov_time * np.sqrt((2 * re_lambda) / (15**0.5 * ao))
    result = get_lagrangian_taylor_microscale_time(
        kolmogorov_time, re_lambda, ao
    )

    assert np.allclose(result, expected, atol=1e-10)


def test_get_lagrangian_taylor_microscale_time_invalid():
    """Test that get_lagrangian_taylor_microscale_time raises errors for invalid
    inputs.
    """
    with pytest.raises(ValueError):
        get_lagrangian_taylor_microscale_time(-0.01, 100, 11)

    with pytest.raises(ValueError):
        get_lagrangian_taylor_microscale_time(0.01, -100, 11)

    with pytest.raises(ValueError):
        get_lagrangian_taylor_microscale_time(0.01, 100, -11)


def test_get_taylor_microscale_scalar():
    """Test get_taylor_microscale with scalar inputs."""
    rms_velocity = 0.5  # m/s
    kinematic_viscosity = 1.5e-5  # m²/s
    turbulent_dissipation = 1e-3  # m²/s³

    expected = rms_velocity * np.sqrt(
        (15 * kinematic_viscosity**2) / turbulent_dissipation
    )
    result = get_taylor_microscale(
        rms_velocity, kinematic_viscosity, turbulent_dissipation
    )

    assert np.isclose(result, expected, atol=1e-10)


def test_get_taylor_microscale_array():
    """Test get_taylor_microscale with NumPy array inputs."""
    rms_velocity = np.array([0.5, 0.8])
    kinematic_viscosity = np.array([1.5e-5, 1.2e-5])
    turbulent_dissipation = np.array([1e-3, 2e-3])

    expected = rms_velocity * np.sqrt(
        (15 * kinematic_viscosity**2) / turbulent_dissipation
    )
    result = get_taylor_microscale(
        rms_velocity, kinematic_viscosity, turbulent_dissipation
    )

    assert np.allclose(result, expected, atol=1e-10)


def test_get_taylor_microscale_invalid():
    """Test that get_taylor_microscale raises errors for invalid inputs."""
    with pytest.raises(ValueError):
        get_taylor_microscale(-0.5, 1.5e-5, 1e-3)

    with pytest.raises(ValueError):
        get_taylor_microscale(0.5, -1.5e-5, 1e-3)

    with pytest.raises(ValueError):
        get_taylor_microscale(0.5, 1.5e-5, -1e-3)


def test_taylor_scales_edge_case():
    """Test both Taylor scale functions with very small values near machine
    precision.
    """
    kolmogorov_time = 1e-10
    re_lambda = 1.0
    ao = 1.0

    expected_tt = kolmogorov_time * np.sqrt((2 * re_lambda) / (15**0.5 * ao))
    result_tt = get_lagrangian_taylor_microscale_time(
        kolmogorov_time, re_lambda, ao
    )

    assert np.isclose(result_tt, expected_tt, atol=1e-10)

    rms_velocity = 1e-10
    kinematic_viscosity = 1e-10
    turbulent_dissipation = 1e-10

    expected_lambda = rms_velocity * np.sqrt(
        (15 * kinematic_viscosity**2) / turbulent_dissipation
    )
    result_lambda = get_taylor_microscale(
        rms_velocity, kinematic_viscosity, turbulent_dissipation
    )

    assert np.isclose(result_lambda, expected_lambda, atol=1e-10)


class TestTaylorMicroscaleReynolds(unittest.TestCase):
    """Unit tests for the Taylor-microscale Reynolds number function."""

    def setUp(self):
        """Set up common test parameters."""
        self.fluid_rms_velocity_scalar = 0.5  # [m/s]
        self.taylor_microscale_scalar = 0.05  # [m]
        self.kinematic_viscosity_scalar = 1.5e-5  # [m²/s]

        self.fluid_rms_velocity_array = np.array([0.5, 1.0, 1.5])  # [m/s]
        self.taylor_microscale_array = np.array([0.05, 0.08, 0.1])  # [m]
        self.kinematic_viscosity_array = np.array(
            [1.5e-5, 1.2e-5, 1.0e-5]
        )  # [m²/s]

    def test_get_taylor_microscale_reynolds_number_scalar(self):
        """Test with scalar inputs."""
        expected = (
            self.fluid_rms_velocity_scalar
            * self.taylor_microscale_scalar
            / self.kinematic_viscosity_scalar
        )

        result = get_taylor_microscale_reynolds_number(
            self.fluid_rms_velocity_scalar,
            self.taylor_microscale_scalar,
            self.kinematic_viscosity_scalar,
        )

        self.assertAlmostEqual(result, expected, places=10)

    def test_get_taylor_microscale_reynolds_number_array(self):
        """Test with NumPy array inputs."""
        expected = (
            self.fluid_rms_velocity_array
            * self.taylor_microscale_array
            / self.kinematic_viscosity_array
        )

        result = get_taylor_microscale_reynolds_number(
            self.fluid_rms_velocity_array,
            self.taylor_microscale_array,
            self.kinematic_viscosity_array,
        )

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_invalid_inputs(self):
        """Ensure validation errors are raised for invalid inputs."""
        with self.assertRaises(ValueError):
            get_taylor_microscale_reynolds_number(
                -0.5, 0.05, 1.5e-5
            )  # Negative fluid RMS velocity

        with self.assertRaises(ValueError):
            get_taylor_microscale_reynolds_number(
                0.5, -0.05, 1.5e-5
            )  # Negative taylor microscale

        with self.assertRaises(ValueError):
            get_taylor_microscale_reynolds_number(
                0.5, 0.05, -1.5e-5
            )  # Negative kinematic viscosity

    def test_edge_cases(self):
        """Test with extreme values (very small and very large)."""
        fluid_rms_velocity = np.array([1e-6, 1e-3, 10.0])  # [m/s]
        taylor_microscale = np.array([1e-6, 1e-3, 10.0])  # [m]
        kinematic_viscosity = np.array([1e-6, 1e-3, 10.0])  # [m²/s]

        result = get_taylor_microscale_reynolds_number(
            fluid_rms_velocity, taylor_microscale, kinematic_viscosity
        )

        self.assertTrue(np.all(np.isfinite(result)))

    def test_consistency(self):
        """Ensure function produces consistent results for repeated inputs."""
        result1 = get_taylor_microscale_reynolds_number(
            self.fluid_rms_velocity_array,
            self.taylor_microscale_array,
            self.kinematic_viscosity_array,
        )

        result2 = get_taylor_microscale_reynolds_number(
            self.fluid_rms_velocity_array,
            self.taylor_microscale_array,
            self.kinematic_viscosity_array,
        )

        np.testing.assert_allclose(result1, result2, atol=1e-10)
