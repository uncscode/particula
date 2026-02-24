"""Test the coagulation kernels for charged particles with calls
via system state.
"""

import numpy as np
import pytest
from particula.dynamics.coagulation.charged_dimensional_kernel import (
    _system_state_properties,
    get_coulomb_kernel_chahl2019_via_system_state,
    get_coulomb_kernel_dyachkov2007_via_system_state,
    get_coulomb_kernel_gatti2008_via_system_state,
    get_coulomb_kernel_gopalakrishnan2012_via_system_state,
    get_hard_sphere_kernel_via_system_state,
)


@pytest.mark.parametrize(
    "kernel_function",
    [
        get_hard_sphere_kernel_via_system_state,
        get_coulomb_kernel_dyachkov2007_via_system_state,
        get_coulomb_kernel_gatti2008_via_system_state,
        get_coulomb_kernel_gopalakrishnan2012_via_system_state,
        get_coulomb_kernel_chahl2019_via_system_state,
    ],
)
def test_dimensioned_coagulation_kernels_array(kernel_function):
    """Test the coagulation kernels for charged particles with
    calls via system state.
    """
    radii = np.array([1e-9, 2e-9, 5e-9]) * 10
    mass = np.array([1e-18, 2e-18, 5e-18])
    charge = np.array([-1, 0, 1])
    temperature = 300.0
    pressure = 101325.0

    kernel_matrix = kernel_function(radii, mass, charge, temperature, pressure)
    assert kernel_matrix.shape == (3, 3)
    assert np.all(np.isfinite(kernel_matrix))


def test_system_state_properties_scalar_outputs():
    """Test scalar inputs return expected shapes and reduced mass."""
    radius = 1.2e-9
    mass = 2.4e-18
    charge = 1.0
    temperature = 300.0
    pressure = 101325.0

    (
        coulomb_potential_ratio,
        diffusive_knudsen,
        sum_of_radii,
        reduced_mass,
        reduced_friction_factor,
    ) = _system_state_properties(
        particle_radius=radius,
        particle_mass=mass,
        particle_charge=charge,
        temperature=temperature,
        pressure=pressure,
    )

    assert coulomb_potential_ratio.shape == (1,)
    assert diffusive_knudsen.shape == (1,)
    assert sum_of_radii.shape == (1, 1)
    assert reduced_mass.shape == (1, 1)
    assert reduced_friction_factor.shape == (1, 1)

    np.testing.assert_allclose(sum_of_radii, np.array([[2 * radius]]))
    np.testing.assert_allclose(reduced_mass, np.array([[mass / 2.0]]))
    assert np.all(np.isfinite(coulomb_potential_ratio))
    assert np.all(np.isfinite(diffusive_knudsen))


def test_system_state_properties_array_outputs():
    """Test array inputs broadcast to square matrices."""
    radius = np.array([1.0e-9, 2.0e-9])
    mass = np.array([1.0e-18, 3.0e-18])
    charge = np.array([0.0, 1.0])
    temperature = 300.0
    pressure = 101325.0

    (
        coulomb_potential_ratio,
        diffusive_knudsen,
        sum_of_radii,
        reduced_mass,
        reduced_friction_factor,
    ) = _system_state_properties(
        particle_radius=radius,
        particle_mass=mass,
        particle_charge=charge,
        temperature=temperature,
        pressure=pressure,
    )

    assert coulomb_potential_ratio.shape == (2, 2)
    assert diffusive_knudsen.shape == (2, 2)
    assert sum_of_radii.shape == (2, 2)
    assert reduced_mass.shape == (2, 2)
    assert reduced_friction_factor.shape == (2, 2)

    expected_sum = np.array(
        [
            [2.0e-9, 3.0e-9],
            [3.0e-9, 4.0e-9],
        ]
    )
    expected_reduced_mass = np.array(
        [
            [0.5e-18, 0.75e-18],
            [0.75e-18, 1.5e-18],
        ]
    )
    np.testing.assert_allclose(sum_of_radii, expected_sum)
    np.testing.assert_allclose(reduced_mass, expected_reduced_mass)
    np.testing.assert_allclose(
        coulomb_potential_ratio, coulomb_potential_ratio.T
    )
    assert np.all(np.isfinite(reduced_friction_factor))
    assert np.all(np.isfinite(diffusive_knudsen))
