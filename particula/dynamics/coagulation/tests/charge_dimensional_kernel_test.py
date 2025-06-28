"""Test the coagulation kernels for charged particles with calls
via system state.
"""

import numpy as np
import pytest

from particula.dynamics.coagulation.charged_dimensional_kernel import (
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
