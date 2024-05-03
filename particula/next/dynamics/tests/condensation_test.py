"""Test the Condensation module."""

import numpy as np
from particula.next.dynamics.condensation import (
    vapor_transition_correction,
    partial_pressure_delta,
    thermal_conductivity,
    first_order_mass_transport_k,
    mass_transfer_rate
)


def test_vapor_transition_correction():
    """Test the vapor_transition_correction function."""
    knudsen_number = 0.5
    mass_accommodation = 0.8
    expected_result = 0.615090213231274
    assert np.isclose(
        vapor_transition_correction(knudsen_number, mass_accommodation),
        expected_result
    )


def test_partial_pressure_delta():
    """Test the partial_pressure_delta function."""
    partial_pressure_gas = 100.0
    partial_pressure_particle = 50.0
    kelvin_term = 0.9
    expected_result = 55.0
    assert np.isclose(
        partial_pressure_delta(
            partial_pressure_gas,
            partial_pressure_particle,
            kelvin_term),
        expected_result
    )


def test_thermal_conductivity():
    """Test the thermal_conductivity function."""
    temperature = 300.0
    expected_result = 0.025689999999999998
    assert np.isclose(
        thermal_conductivity(temperature),
        expected_result
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
