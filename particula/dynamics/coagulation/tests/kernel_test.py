"""Test Kernel class and its methods."""

import numpy as np
from particula.dynamics.coagulation import kernel

# Define constants for common test data

DIFFUSIVE_KNUDSEN = np.array([0.5, 1.0, 5.0, 10.0])
COULOMB_POTENTIAL_RATIO_ARRAY = np.array([0.7, 0.9, 1.0, 1.1])


def test_kernel_call():
    """Test calling the kernel method, common for all kernel strategies."""
    dimensionless_kernel_in = np.array([[1.0, 2.0], [3.0, 4.0]])
    coulomb_potential_ratio_in = np.array([[0.5, 0.6], [0.7, 0.8]])
    sum_of_radii_in = np.array([[1e-9, 2e-9], [3e-9, 4e-9]])
    reduced_mass_in = np.array([[1e-18, 2e-18], [3e-18, 4e-18]])
    reduced_friction_factor_in = np.array([[1.0, 1.5], [2.0, 2.5]])
    # test kernel dimensioned
    kernel_concrete = kernel.HardSphere()
    dimension_result = kernel_concrete.kernel(
        dimensionless_kernel=dimensionless_kernel_in,
        coulomb_potential_ratio=coulomb_potential_ratio_in,
        sum_of_radii=sum_of_radii_in,
        reduced_mass=reduced_mass_in,
        reduced_friction_factor=reduced_friction_factor_in,
    )
    expected_result = np.array(
        [[1.77061203e-09, 2.31008442e-08], [1.12232711e-07, 3.56834831e-07]]
    )
    np.testing.assert_almost_equal(
        dimension_result, expected_result, decimal=4
    )


def test_hard_sphere():
    """
    Test the hard_sphere function with a single value and
    array of diffusive_knudsen values.
    """
    # dimensionless
    kernel_concrete = kernel.HardSphere()
    dimensionless_result = kernel_concrete.dimensionless(
        DIFFUSIVE_KNUDSEN, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    expected_result = np.array(
        [1.65971644, 4.12694075, 24.16690909, 49.22484307]
    )
    np.testing.assert_almost_equal(
        dimensionless_result, expected_result, decimal=4
    )


def test_coulomb_dyachkov2007():
    """
    Test the coulomb_dyachkov2007 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # dimensionless
    kernel_concrete = kernel.CoulombDyachkov2007()
    dimensionless_result = kernel_concrete.dimensionless(
        DIFFUSIVE_KNUDSEN, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    expected_result = np.array(
        [1.73703563, 4.60921277, 26.22159795, 51.92102133]
    )
    np.testing.assert_almost_equal(
        dimensionless_result, expected_result, decimal=4
    )


def test_coulomb_gatti2008():
    """
    Test the coulomb_gatti2008 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # dimensionless
    kernel_concrete = kernel.CoulombGatti2008()
    dimensionless_result = kernel_concrete.dimensionless(
        DIFFUSIVE_KNUDSEN, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    expected_result = np.array(
        [2.00132915, 5.10865767, 26.42422258, 52.43789491]
    )
    np.testing.assert_almost_equal(
        dimensionless_result, expected_result, decimal=4
    )


def test_coulomb_gopalakrishnan2012():
    """
    Test the coulomb_gopalakrishnan2012 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # dimensionless
    kernel_concrete = kernel.CoulombGopalakrishnan2012()
    dimensionless_result = kernel_concrete.dimensionless(
        DIFFUSIVE_KNUDSEN, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    expected_result = np.array(
        [1.83746548, 4.83694019, 24.16690909, 49.22484307]
    )
    np.testing.assert_almost_equal(
        dimensionless_result, expected_result, decimal=4
    )


def test_coulomb_chahl2019():
    """
    Test the coulomb_chahl2019 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # dimensionless
    kernel_concrete = kernel.CoulumbChahl2019()
    dimensionless_result = kernel_concrete.dimensionless(
        DIFFUSIVE_KNUDSEN, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    expected_result = np.array(
        [1.65863442, 4.37444613, 28.05501739, 59.74082667]
    )
    np.testing.assert_almost_equal(
        dimensionless_result, expected_result, decimal=4
    )
