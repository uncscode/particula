"""Test the Condensation module."""

import numpy as np
from particula.next.dynamics.condensation import (
    first_order_mass_transport_k,
    mass_transfer_rate
)


def test_first_order_mass_transport_k():
    """Test the first_order_mass_transport_k function."""
    radius = 1e-6
    vapor_transition = 0.6
    diffusion_coefficient = 2e-9
    expected_result = 1.0053096491487334e-17
    assert np.isclose(
        first_order_mass_transport_k(
            radius, vapor_transition, diffusion_coefficient),
        expected_result
    )


def test_mass_transfer_rate():
    """Test the mass_transfer_rate function."""
    pressure_delta = 10.0
    first_order_mass_transport = 1e-17
    temperature = 300.0
    molar_mass = 0.02897
    expected_result = 1.16143004e-21
    assert np.isclose(
        mass_transfer_rate(
            pressure_delta,
            first_order_mass_transport,
            temperature,
            molar_mass),
        expected_result
    )
