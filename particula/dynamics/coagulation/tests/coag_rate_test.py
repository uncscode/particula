"""Test rate calculations for coagulation dynamics.

tested for evaluation not accuracy of the calculations."""

import numpy as np
from particula.dynamics.coagulation import rate

# Define constants for test data
# Define constants for test data
CONCENTRATION = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
KERNEL = np.array(
    [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
        [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1],
        [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8],
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        [0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2],
        [0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9],
    ]
)
RADIUS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])


def test_discrete_loss():
    """
    Test the discrete_loss function with predefined concentration and kernel
    values.
    """
    result = rate.discrete_loss(CONCENTRATION, KERNEL)
    expected = np.array([14.0, 56.0, 126.0, 224.0, 350.0, 504.0, 686.0])
    np.testing.assert_almost_equal(result, expected, decimal=6)


def test_discrete_gain():
    """
    Test the discrete_gain function with predefined concentration and kernel
    values.
    """
    result = rate.discrete_gain(RADIUS, CONCENTRATION, KERNEL)
    expected = np.array(
        [
            8.88984299e-02,
            8.07430616e-01,
            4.71791309e00,
            1.83365875e01,
            5.45636475e01,
            1.33885983e02,
            2.86922649e02,
        ]
    )
    np.testing.assert_almost_equal(result, expected, decimal=6)


def test_continuous_loss():
    """
    Test the continuous_loss function with predefined radius, concentration,
    and kernel values.
    """
    result = rate.continuous_loss(RADIUS, CONCENTRATION, KERNEL)
    expected = np.array([1.15, 4.6, 10.35, 18.4, 28.75, 41.4, 56.35])
    np.testing.assert_almost_equal(result, expected, decimal=6)


def test_continuous_gain():
    """
    Test the continuous_gain function with predefined radius, concentration,
    and kernel values.
    """
    result = rate.continuous_gain(RADIUS, CONCENTRATION, KERNEL)
    expected = np.array(
        [
            8.88984299e-03,
            8.07430616e-02,
            4.71791309e-01,
            1.83365875e00,
            5.45636475e00,
            1.33885983e01,
            2.86922649e01,
        ]
    )
    np.testing.assert_almost_equal(result, expected, decimal=6)
