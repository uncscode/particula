"""
Test the turbulent shear kernel functions.
"""

import numpy as np
import pytest
from particula.dynamics.coagulation.turbulent_shear_kernel import (
    saffman_turner_1956,
    saffman_turner_1956_via_system_state,
)
from particula.gas.properties.kinematic_viscosity import (
    get_kinematic_viscosity_via_system_state,
)


def test_turbulent_shear_kernel_single_value():
    """
    Test turbulent_shear_kernel with single float inputs for particle
    diameters, turbulent kinetic energy, and kinematic viscosity.
    """
    diameter_particle = np.array([1e-6, 2e-6])  # example diameters [m]
    turbulent_kinetic_energy = (
        1.0e-4  # example turbulent kinetic energy [m^2/s^2]
    )
    kinematic_viscosity = 1.5e-5  # example kinematic viscosity [m^2/s]
    expected_kernel = (
        np.pi * turbulent_kinetic_energy / (120 * kinematic_viscosity)
    ) ** 0.5 * (
        diameter_particle[:, np.newaxis] + diameter_particle[np.newaxis, :]
    ) ** 3
    value = saffman_turner_1956(
        diameter_particle/2, turbulent_kinetic_energy, kinematic_viscosity
    )
    np.testing.assert_allclose(value, expected_kernel, rtol=1e-6)


def test_turbulent_shear_kernel_via_system_state():
    """
    Test turbulent_shear_kernel_via_system_state with system state inputs.
    """
    particle_radius = np.array([1e-6, 2e-6])  # example diameters [m]
    turbulent_kinetic_energy = (
        1.0e-4  # example turbulent kinetic energy [m^2/s^2]
    )
    temperature = 298  # example temperature [K]
    fluid_density = 1.2  # example fluid density [kg/m^3]
    kinematic_viscosity = get_kinematic_viscosity_via_system_state(
        temperature=temperature, fluid_density=fluid_density
    )
    expected_kernel = saffman_turner_1956(
        particle_radius, turbulent_kinetic_energy, kinematic_viscosity
    )
    value = saffman_turner_1956_via_system_state(
        particle_radius, turbulent_kinetic_energy, temperature, fluid_density
    )
    np.testing.assert_allclose(value, expected_kernel, rtol=1e-6)


def test_turbulent_shear_kernel_input_validation():
    """
    Ensure that providing incorrect input types to turbulent_shear_kernel
    raises a TypeError.
    """
    with pytest.raises(TypeError):
        saffman_turner_1956("not a number", "not a number", "not a number")


def test_turbulent_shear_kernel_via_system_state_input_validation():
    """
    Ensure that providing incorrect input types to
    turbulent_shear_kernel_via_system_state raises a TypeError.
    """
    with pytest.raises(TypeError):
        saffman_turner_1956_via_system_state(
            "not a number", "not a number", "not a number", "not a number"
        )
