"""
Test the wall loss functions.
"""

import numpy as np

from particula import u
from particula.util.input_handling import in_handling
from particula.util.debye_function import df1
from particula.util.settling_velocity import psv
from particula.util.diffusion_coefficient import pdc
from particula.util.wall_loss import (
    spherical_wall_loss_coefficient,
    rectangle_wall_loss,
    wlc)


def test_spherical_wall_loss_coefficient():
    # Test case with hypothetical values
    ktp_value = 1.5  # Example value
    diffusion_coefficient_value = 0.01  # Example value
    settling_velocity_value = 0.05  # Example value
    chamber_radius = 2.0  # Example value

    expected_output = 0.11816  # This is a hypothetical value.

    # Call the function with the test case
    calculated_output = spherical_wall_loss_coefficient(
        ktp_value,
        diffusion_coefficient_value,
        settling_velocity_value,
        chamber_radius
    )

    # Assertion to check if the calculated output matches the expected output
    assert np.isclose(calculated_output, expected_output, rtol=1e-4)


def test_rectangle_wall_loss():
    # Test case with hypothetical values
    ktp_value = 1.5  # Example value
    diffusion_coefficient_value = 0.01  # Example value
    settling_velocity_value = 0.05  # Example value
    dimension = [1, 1, 1]  # Example value

    expected_output = 0.47312  # This is a hypothetical value.

    # Call the function with the test case
    calculated_output = rectangle_wall_loss(
        ktp_value,
        diffusion_coefficient_value,
        settling_velocity_value,
        dimension
    )

    # Assertion to check if the calculated output matches the expected output
    assert np.isclose(calculated_output, expected_output, rtol=1e-4)