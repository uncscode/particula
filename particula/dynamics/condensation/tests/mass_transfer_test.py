"""Test the Condensation module."""

import numpy as np
from particula.dynamics.condensation.mass_transfer import (
    first_order_mass_transport_k,
    mass_transfer_rate,
    calculate_mass_transfer,
    calculate_mass_transfer_single_species,
    calculate_mass_transfer_multiple_species,
)


def test_first_order_mass_transport_k():
    """Test the first_order_mass_transport_k function."""
    radius = 1e-6
    vapor_transition = 0.6
    diffusion_coefficient = 2e-9
    expected_result = 1.5079644737231005e-14
    result = first_order_mass_transport_k(
        radius, vapor_transition, diffusion_coefficient
    )
    np.testing.assert_allclose(result, expected_result, rtol=1e-8)


def test_multi_radius_first_order_mass_transport_k():
    """Test the first_order_mass_transport_k function for multiple radii."""
    radius = np.array([1e-6, 2e-6, 3e-6])
    vapor_transition = 0.6
    diffusion_coefficient = 2e-9
    expected_result = np.array(
        [1.50796447e-14, 3.01592895e-14, 4.52389342e-14]
    )
    result = first_order_mass_transport_k(
        radius, vapor_transition, diffusion_coefficient
    )
    np.testing.assert_allclose(result, expected_result, rtol=1e-8)


def test_mass_transfer_rate():
    """Test the mass_transfer_rate function."""
    pressure_delta = 10.0
    first_order_mass_transport = 1e-17
    temperature = 300.0
    molar_mass = 0.02897
    expected_result = 1.16143004e-21
    assert np.isclose(
        mass_transfer_rate(
            pressure_delta, first_order_mass_transport, temperature, molar_mass
        ),
        expected_result,
    )


def test_mass_transfer_mulit_particle_rate():
    """Test the mass_transfer_rate multi radii function."""
    pressure_delta = np.array([10.0, 15.0])
    first_order_mass_transport = np.array([1e-17, 2e-17])
    temperature = 300.0
    molar_mass = 0.02897
    expected_result = np.array([1.16143004e-21, 3.48429013e-21])
    result = mass_transfer_rate(
        pressure_delta, first_order_mass_transport, temperature, molar_mass
    )
    np.testing.assert_allclose(result, expected_result, rtol=1e-8)


def test_multi_species_mass_transfer_rate():
    """Test the mass_transfer_rate function for multiple species."""
    pressure_delta = np.array([10.0, 15.0])
    first_order_mass_transport = np.array([1e-17, 2e-17])
    temperature = 300.0
    molar_mass = np.array([0.02897, 0.018015])
    expected_result = np.array([1.16143004e-21, 2.16670648e-21])
    result = mass_transfer_rate(
        pressure_delta, first_order_mass_transport, temperature, molar_mass
    )
    np.testing.assert_allclose(result, expected_result, rtol=1e-8)


def test_single_species_condensation_not_enough_gas_mass():
    """Test mass transfer for a single particle species where there is not
    enough gas mass for full condensation."""
    mass_rate = np.array([0.1, 0.5])  # kg/s (mass transfer rate per particle)
    time_step = 10  # seconds
    gas_mass = np.array([0.5])  # kg (not enough to satisfy both requests)
    particle_mass = np.array([1.0, 50])  # kg
    particle_concentration = np.array([1, 0.5])  # particles/m^3

    # Calculate the total mass to be transferred, accounting for particle
    # concentration
    total_mass_to_change = (
        mass_rate * time_step * particle_concentration
    )  # Total mass requested

    # Total requested mass is (0.1 * 10 * 1) + (0.5 * 10 * 0.5) = 1.0 kg,
    # but only 0.5 kg is available
    # Scaling factor: we need to scale the mass transfer so that the total
    # matches the available gas mass
    scaling_factor = gas_mass / total_mass_to_change.sum()

    # Expected mass transfer is the scaled version of the mass_to_change
    expected_mass_transfer = total_mass_to_change * scaling_factor

    # Calculate using the direct single species function
    result_direct = calculate_mass_transfer_single_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(
        result_direct, expected_mass_transfer, rtol=1e-8
    )
    # second calc
    result_direct2 = calculate_mass_transfer_single_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(
        result_direct2, expected_mass_transfer, rtol=1e-8
    )

    # Calculate using the general helper function
    result = calculate_mass_transfer(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(result, expected_mass_transfer, rtol=1e-8)


def test_single_species_evaporation_not_enough_particle_mass():
    """Test mass transfer for a single particle species where there is not
    enough particle mass for full evaporation."""
    mass_rate = np.array([-0.2, -8])  # kg/s (negative for evaporation)
    time_step = 10  # seconds
    gas_mass = np.array([1.0])  # kg
    particle_mass = np.array([0.8, 0.3])  # kg per particle
    particle_concentration = np.array([1, 5])  # particles/m^3

    # Total available particle mass is particle_mass * particle_concentration
    # particle_mass * particle_concentration = [0.8 * 1, 0.3 * 5] = [0.8, 1.5]
    # Requested mass transfer is [-0.2 * 10, -8 * 10] = [-2.0, -80.0]
    # However, the transfer is limited by available particle mass: [-0.8, -1.5]
    expected_mass_transfer = np.array([-0.8, -1.5])

    result_direct = calculate_mass_transfer_single_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(
        result_direct, expected_mass_transfer, rtol=1e-8
    )

    result = calculate_mass_transfer(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(result, expected_mass_transfer, rtol=1e-8)


def test_multiple_species_condensation():
    """
    Test mass transfer for multiple particle and gas species (n=2, m=3)
    where there is not enough gas mass for full condensation, and particle
    concentration is greater than 1.
    """
    # mass_rate for 2 particles and 3 gas species
    mass_rate = np.array([[0.1, 0.05, 0.03], [0.2, 0.15, 0.07]])  # kg/s
    time_step = 10  # seconds

    # gas_mass for 3 gas species
    gas_mass = np.array([1.0, 0.8, 0.5])  # kg

    # particle_mass for 2 particles and 3 gas species
    particle_mass = np.array([[1.0, 0.9, 0.8], [1.2, 1.0, 0.7]])  # kg

    # particle_concentration for 2 particles, greater than 1
    particle_concentration = np.array([5, 4])  # particles/m^3

    # Step 1: Calculate the total mass to change (before scaling)
    mass_to_change = (
        mass_rate * time_step * particle_concentration[:, np.newaxis]
    )

    # Step 2: Calculate the total requested mass for each gas species
    total_requested_mass = mass_to_change.sum(axis=0)

    # Step 3: Apply scaling if requested mass exceeds available gas mass
    scaling_factor = np.ones_like(gas_mass)
    scaling_mask = total_requested_mass > gas_mass
    scaling_factor[scaling_mask] = (
        gas_mass[scaling_mask] / total_requested_mass[scaling_mask]
    )

    # Step 4: Calculate expected mass transfer (scaled)
    expected_mass_transfer = mass_to_change * scaling_factor

    # Test the direct multiple species function
    result_direct = calculate_mass_transfer_multiple_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    # Check that the total mass transfer for each gas species is equal to the
    # total mass transfer for the gas phase
    np.testing.assert_allclose(result_direct.sum(axis=0), gas_mass, rtol=1e-8)
    # Check for each individual particle and species
    np.testing.assert_allclose(
        result_direct, expected_mass_transfer, rtol=1e-8
    )

    # Test the general helper function
    result = calculate_mass_transfer(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(result, expected_mass_transfer, rtol=1e-8)


def test_multiple_species_evaporation_not_enough_particle_mass():
    """
    Test mass transfer for multiple particle and gas species (n=2, m=3)
    where the particle mass is insufficient for full evaporation, and
    particle concentration is greater than 1.
    """
    # mass_rate for 2 particles and 3 gas species (negative for evaporation)
    mass_rate = np.array([[-0.1, -0.5, -0.3], [-0.2, -0.15, -0.7]])  # kg/s
    time_step = 10  # seconds

    # gas_mass for 3 gas species (not relevant for evaporation in this case)
    gas_mass = np.array([1.0, 0.8, 0.5])  # kg

    # particle_mass for 2 particles and 3 gas species
    particle_mass = np.array([[0.7, 0.9, 0.8], [0.8, 0.5, 0.6]])  # kg

    # particle_concentration for 2 particles, greater than 1
    particle_concentration = np.array([10, 10])  # particles/m^3

    # Step 1: Calculate the total mass to change for evaporation
    mass_to_change = (
        mass_rate * time_step * particle_concentration[:, np.newaxis]
    )

    # Step 2: Calculate the available particle mass for evaporation
    # Available mass = particle_mass * particle_concentration
    available_particle_mass = (
        particle_mass * particle_concentration[:, np.newaxis]
    )

    # Step 3: Limit evaporation by available particle mass
    expected_mass_transfer = np.maximum(
        mass_to_change, -available_particle_mass
    )

    # Test the direct multiple species function
    result_direct = calculate_mass_transfer_multiple_species(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    # Check the individual mass transfer for each particle and species
    np.testing.assert_allclose(
        result_direct, expected_mass_transfer, rtol=1e-8
    )
    # Check the total mass transfer for each particle
    total_mass_possible = np.sum(available_particle_mass, axis=0)
    np.testing.assert_allclose(
        np.sum(result_direct, axis=0), -1 * total_mass_possible, rtol=1e-8
    )

    # Test the general helper function
    result = calculate_mass_transfer(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_allclose(result, expected_mass_transfer, rtol=1e-8)


def test_zero_mass_transfer():
    """Test mass transfer when there is no mass transfer."""
    mass_rate = np.array([0.0])  # No mass transfer
    time_step = 10  # seconds
    gas_mass = np.array([1.0])  # kg
    particle_mass = np.array([1.0])  # kg
    particle_concentration = np.array([1])  # particles/m^3

    expected_mass_transfer = np.array([0.0])  # No mass transfer should occur

    result = calculate_mass_transfer(
        mass_rate, time_step, gas_mass, particle_mass, particle_concentration
    )
    np.testing.assert_array_almost_equal(result, expected_mass_transfer)
