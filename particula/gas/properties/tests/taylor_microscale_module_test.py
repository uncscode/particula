"""
Test module for the Taylor microscale functions.
"""

import pytest
import numpy as np
from particula.gas.properties.taylor_microscale_module import (
    get_lagrangian_taylor_microscale_time,
    get_taylor_microscale,
)


def test_get_lagrangian_taylor_microscale_time_scalar():
    """
    Test get_lagrangian_taylor_microscale_time with scalar inputs.
    """
    kolmogorov_time = 0.01  # s
    re_lambda = 100  # Reynolds number
    ao = 11  # Normalized acceleration variance

    expected = kolmogorov_time * np.sqrt((2 * re_lambda) / (15**0.5 * ao))
    result = get_lagrangian_taylor_microscale_time(
        kolmogorov_time, re_lambda, ao
    )

    assert np.isclose(result, expected, atol=1e-10)


def test_get_lagrangian_taylor_microscale_time_array():
    """
    Test get_lagrangian_taylor_microscale_time with NumPy array inputs.
    """
    kolmogorov_time = np.array([0.01, 0.02])
    re_lambda = np.array([100, 200])
    ao = np.array([11, 12])

    expected = kolmogorov_time * np.sqrt((2 * re_lambda) / (15**0.5 * ao))
    result = get_lagrangian_taylor_microscale_time(
        kolmogorov_time, re_lambda, ao
    )

    assert np.allclose(result, expected, atol=1e-10)


def test_get_lagrangian_taylor_microscale_time_invalid():
    """
    Test that get_lagrangian_taylor_microscale_time raises errors for invalid
    inputs.
    """
    with pytest.raises(ValueError):
        get_lagrangian_taylor_microscale_time(-0.01, 100, 11)

    with pytest.raises(ValueError):
        get_lagrangian_taylor_microscale_time(0.01, -100, 11)

    with pytest.raises(ValueError):
        get_lagrangian_taylor_microscale_time(0.01, 100, -11)


def test_get_taylor_microscale_scalar():
    """
    Test get_taylor_microscale with scalar inputs.
    """
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
    """
    Test get_taylor_microscale with NumPy array inputs.
    """
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
    """
    Test that get_taylor_microscale raises errors for invalid inputs.
    """
    with pytest.raises(ValueError):
        get_taylor_microscale(-0.5, 1.5e-5, 1e-3)

    with pytest.raises(ValueError):
        get_taylor_microscale(0.5, -1.5e-5, 1e-3)

    with pytest.raises(ValueError):
        get_taylor_microscale(0.5, 1.5e-5, -1e-3)


def test_taylor_scales_edge_case():
    """
    Test both Taylor scale functions with very small values near machine
    precision.
    """
    kolmogorov_time = 1e-10
    re_lambda = 1.0
    ao = 1.0

    expected_TT = kolmogorov_time * np.sqrt((2 * re_lambda) / (15**0.5 * ao))
    result_TT = get_lagrangian_taylor_microscale_time(
        kolmogorov_time, re_lambda, ao
    )

    assert np.isclose(result_TT, expected_TT, atol=1e-10)

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
