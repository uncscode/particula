"""
Test the normalize_accel_variance function.
"""

import pytest
import numpy as np

from particula.gas.properties.normalize_accel_variance import (
    get_normalized_accel_variance_ao2008,
)


def test_get_normalized_accel_variance_scalar():
    """
    Test get_normalized_accel_variance with scalar inputs.
    """
    re_lambda = 100  # Taylor-microscale Reynolds number

    expected = (11 + 7 * re_lambda) / (205 + re_lambda)
    result = get_normalized_accel_variance_ao2008(re_lambda)

    assert np.isclose(result, expected, atol=1e-10)


def test_get_normalized_accel_variance_array():
    """
    Test get_normalized_accel_variance with NumPy array inputs.
    """
    re_lambda = np.array([50, 150, 300])

    expected = (11 + 7 * re_lambda) / (205 + re_lambda)
    result = get_normalized_accel_variance_ao2008(re_lambda)

    assert np.allclose(result, expected, atol=1e-10)


def test_get_normalized_accel_variance_invalid():
    """
    Test that get_normalized_accel_variance raises errors for invalid inputs.
    """
    with pytest.raises(ValueError):
        get_normalized_accel_variance_ao2008(-10)  # Negative Reynolds number

    with pytest.raises(ValueError):
        get_normalized_accel_variance_ao2008(
            np.array([-10, 50])
        )  # Array with negative value

    with pytest.raises(ValueError):
        get_normalized_accel_variance_ao2008(
            0
        )  # Zero Reynolds number (should be positive)


def test_get_normalized_accel_variance_edge_case():
    """
    Test get_normalized_accel_variance with very small values near machine precision.
    """
    re_lambda = 1e-10

    expected = (11 + 7 * re_lambda) / (205 + re_lambda)
    result = get_normalized_accel_variance_ao2008(re_lambda)

    assert np.isclose(result, expected, atol=1e-10)
