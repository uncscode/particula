"""Test Brownian module functions.
"""

import pytest
import numpy as np
from particula.dynamics.coagulation import brownian_kernel
from particula.constants import BOLTZMANN_CONSTANT


def test_mean_free_path_l_single_value():
    """
    Test mean_free_path_l with single float inputs for diffusivity and mean
    thermal speed.
    """
    diffusivity_particle = 1.0e-5  # example diffusivity [m^2/s]
    mean_thermal_speed_particle = 100  # example speed [m/s]
    expected_path = (
        8 * diffusivity_particle / (np.pi * mean_thermal_speed_particle)
    )
    value = brownian_kernel.mean_free_path_l(
        diffusivity_particle, mean_thermal_speed_particle
    )
    assert np.isclose(value, expected_path)


def test_mean_free_path_l_array_input():
    """
    Test mean_free_path_l with numpy array inputs for diffusivity and mean
    thermal speed.
    """
    diffusivity_particle = np.array([1.0e-5, 1.0e-4])
    mean_thermal_speed_particle = np.array([100, 200])
    expected_path = (
        8 * diffusivity_particle / (np.pi * mean_thermal_speed_particle)
    )
    value = brownian_kernel.mean_free_path_l(
        diffusivity_particle, mean_thermal_speed_particle
    )
    np.testing.assert_allclose(value, expected_path, rtol=1e-6)


def test_mean_free_path_l_input_validation():
    """
    Ensure that providing incorrect input types to mean_free_path_l raises a
    TypeError.
    """
    with pytest.raises(TypeError):
        brownian_kernel.mean_free_path_l("not a number", "also not a number")


def test_g_collection_term_single_value():
    """
    Test g_collection_term with single float inputs for mean free path and
    radius of particles.
    """
    mean_free_path_particle = 6.085302617379366e-08  # mean free path in meters
    radius_particle = 1e-5  # radius in meters
    expected_value = (
        (2 * radius_particle + mean_free_path_particle) ** 3
        - (4 * radius_particle**2 + mean_free_path_particle**2) ** (3 / 2)
    ) / (6 * radius_particle * mean_free_path_particle) - 2 * radius_particle
    value = brownian_kernel.g_collection_term(
        mean_free_path_particle, radius_particle
    )
    assert np.isclose(value, expected_value)


def test_g_collection_term_array_input():
    """
    Test g_collection_term with numpy array inputs for mean free path and
    radius of particles.
    """
    mean_free_path_particle = np.array([0.0005, 0.0005])
    radius_particle = np.array([0.0001, 0.0002])
    expected_value = (
        (2 * radius_particle + mean_free_path_particle) ** 3
        - (4 * radius_particle**2 + mean_free_path_particle**2) ** (3 / 2)
    ) / (6 * radius_particle * mean_free_path_particle) - 2 * radius_particle
    value = brownian_kernel.g_collection_term(
        mean_free_path_particle, radius_particle
    )
    np.testing.assert_allclose(value, expected_value, rtol=1e-6)


def test_g_collection_term_input_validation():
    """
    Ensure that providing incorrect input types to g_collection_term raises
    a TypeError.
    """
    with pytest.raises(TypeError):
        brownian_kernel.g_collection_term("not a number", "also not a number")


def test_g_collection_term_zero_radius():
    """
    Test that g_collection_term handles a case where the radius is zero
    without crashing.
    """
    mean_free_path_particle = 0.0005
    radius_particle = 0.0
    with pytest.raises(ZeroDivisionError):
        brownian_kernel.g_collection_term(
            mean_free_path_particle, radius_particle
        )


def test_brownian_diffusivity_single_value():
    """
    Test brownian_diffusivity with single float inputs for temperature and
    aerodynamic mobility.
    """
    temperature = 298  # example temperature in Kelvin
    aerodynamic_mobility = 1.5e-9  # example mobility in m^2/s
    expected_diffusivity = (
        float(BOLTZMANN_CONSTANT.m) * temperature * aerodynamic_mobility
    )
    value = brownian_kernel.brownian_diffusivity(
        temperature, aerodynamic_mobility
    )
    assert np.isclose(value, expected_diffusivity)


def test_brownian_diffusivity_array_input():
    """
    Test brownian_diffusivity with numpy array inputs for temperature and
    aerodynamic mobility.
    """
    temperature = np.array([298, 300])
    aerodynamic_mobility = np.array([1.5e-9, 2.0e-9])
    expected_diffusivity = (
        float(BOLTZMANN_CONSTANT.m) * temperature * aerodynamic_mobility
    )
    value = brownian_kernel.brownian_diffusivity(
        temperature, aerodynamic_mobility
    )
    np.testing.assert_allclose(value, expected_diffusivity, rtol=1e-6)


def test_brownian_coagulation_kernel_basic():
    """
    Test brownian_coagulation_kernel with basic input values.
    """
    radius_particle = np.array([1e-9, 2e-9])  # radii in meters
    diffusivity_particle = np.array([1e-12, 1e-12])  # diffusivity in m^2/s
    g_collection_term_particle = np.array([0.5, 0.5])  # dimensionless
    mean_thermal_speed_particle = np.array([100, 100])  # speed in m/s
    alpha_collision_efficiency = np.array([1.0, 1.0])  # dimensionless

    # Calculate expected values
    expected_kernel = np.array(
        [[1.77697548e-15, 3.99769516e-15], [3.99769516e-15, 7.10577039e-15]]
    )

    result_kernel = brownian_kernel.brownian_coagulation_kernel(
        radius_particle,
        diffusivity_particle,
        g_collection_term_particle,
        mean_thermal_speed_particle,
        alpha_collision_efficiency,
    )

    np.testing.assert_allclose(result_kernel, expected_kernel, rtol=1e-6)


def test_brownian_coagulation_kernel_with_defaults():
    """
    Test that default parameters are handled correctly.
    """
    radius_particle = np.array([1e-9])  # single radius
    diffusivity_particle = np.array([1e-12])  # single diffusivity
    g_collection_term_particle = np.array([0.5])  # single collection term
    mean_thermal_speed_particle = np.array([100])  # single speed

    # expected result for a single input with default alpha
    expected_kernel = np.array([1.77697548e-15])

    result_kernel = brownian_kernel.brownian_coagulation_kernel(
        radius_particle,
        diffusivity_particle,
        g_collection_term_particle,
        mean_thermal_speed_particle,
    )
    assert np.isclose(result_kernel, expected_kernel)


def test_brownian_coagulation_kernel_input_validation():
    """
    Ensure that providing incorrect input types raises a TypeError.
    """
    with pytest.raises(TypeError):
        brownian_kernel.brownian_coagulation_kernel(
            "not a number", "not a number", "not a number", "not a number"
        )


def test_brownian_coagulation_kernel_via_system_state_basic():
    """
    Test the complete system function with basic input values.
    """
    # diameters, 2 nm, 1 um, 20 um
    radius_particle = np.array([1e-9, 5e-7, 10e-6])  # radii in meters
    mass_particle = 4 / 3 * np.pi * radius_particle**3 * 1000  # mass in kg
    temperature = 298  # temperature in Kelvin
    pressure = 101325  # pressure in Pascal
    alpha_collision_efficiency = np.array([1.0, 1.0, 1.0])  # dimensionless

    # Seinfeld and Pandis Table 13.3, larger particles seem off by 2x
    # table 13.3
    # [8.9e-16, 7.8e-12, 17e-9],
    # [7.8e-12, 6.9e-16, 3.9e-15],
    # [17e-9, 3.9e-15, 6.0e-16]
    expected_output = np.array(
        [
            [8.88541843e-16, 7.57666671e-12, 1.64739803e-10],
            [7.57666671e-12, 6.80940034e-16, 3.82064762e-15],
            [1.64739803e-10, 3.82064762e-15, 5.99655468e-16],
        ]
    )

    # Call the system function
    result = brownian_kernel.brownian_coagulation_kernel_via_system_state(
        radius_particle,
        mass_particle,
        temperature,
        pressure,
        alpha_collision_efficiency,
    )

    # Assert to check if the calculated results match expected results closely
    np.testing.assert_allclose(result, expected_output, rtol=1e-6)


def test_brownian_coagulation_kernel_via_system_state_input_validation():
    """
    Ensure that providing incorrect input types or sizes raises appropriate
    errors.
    """
    with pytest.raises(TypeError):
        brownian_kernel.brownian_coagulation_kernel_via_system_state(
            radius_particle="not a number",
            mass_particle="not a number",
            temperature="not a number",
            pressure="not a number",
            alpha_collision_efficiency="not a number",
        )  # pytype: disable=wrong-arg-types
