"""Test for transition_regime calculation functions."""

import numpy as np
import pytest

from particula.dynamics.coagulation import charged_dimensionless_kernel

# test single value
DIFFUSIVE_KNUDSEN_SINGLE = 0.1
COULOMB_POTENTIAL_RATIO_SINGLE = 0.5

# test array
DIFFUSIVE_KNUDSEN_ARRAY = np.array([0.5, 1.0, 5.0, 10.0])
COULOMB_POTENTIAL_RATIO_ARRAY = np.array([0.7, 0.9, 1.0, 1.1])


def test_hard_sphere():
    """Test the hard_sphere function with a single value and
    array of diffusive_knudsen values.
    """
    # single value
    expected_single = 0.10960430161885967
    result_single = charged_dimensionless_kernel.get_hard_sphere_kernel(
        DIFFUSIVE_KNUDSEN_SINGLE
    )
    np.testing.assert_almost_equal(result_single, expected_single, decimal=4)
    # array
    expected = np.array([1.65971644, 4.12694075, 24.16690909, 49.22484307])
    result = charged_dimensionless_kernel.get_hard_sphere_kernel(
        DIFFUSIVE_KNUDSEN_ARRAY
    )
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_coulomb_dyachkov2007():
    """Test the coulomb_dyachkov2007 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # single value
    expected_single = 0.10960430161885967
    result_single = (
        charged_dimensionless_kernel.get_coulomb_kernel_dyachkov2007(
            DIFFUSIVE_KNUDSEN_SINGLE, COULOMB_POTENTIAL_RATIO_SINGLE
        )
    )
    np.testing.assert_almost_equal(result_single, expected_single, decimal=4)
    # array
    expected = np.array([1.73703563, 4.60921277, 26.22159795, 51.92102133])
    result = charged_dimensionless_kernel.get_coulomb_kernel_dyachkov2007(
        DIFFUSIVE_KNUDSEN_ARRAY, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_coulomb_gatti2008():
    """Test the coulomb_gatti2008 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # single value
    expected_single = 0.12621027
    result_single = charged_dimensionless_kernel.get_coulomb_kernel_gatti2008(
        DIFFUSIVE_KNUDSEN_SINGLE, COULOMB_POTENTIAL_RATIO_SINGLE
    )
    np.testing.assert_almost_equal(result_single, expected_single, decimal=4)
    # array
    expected = np.array([2.00132915, 5.10865767, 26.42422258, 52.43789491])
    result = charged_dimensionless_kernel.get_coulomb_kernel_gatti2008(
        DIFFUSIVE_KNUDSEN_ARRAY, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_coulomb_gopalakrishnan2012():
    """Test the coulomb_gopalakrishnan2012 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # single value
    expected_single = 0.1096043
    result_single = (
        charged_dimensionless_kernel.get_coulomb_kernel_gopalakrishnan2012(
            DIFFUSIVE_KNUDSEN_SINGLE, COULOMB_POTENTIAL_RATIO_SINGLE
        )
    )
    np.testing.assert_almost_equal(result_single, expected_single, decimal=4)
    # array
    expected = np.array([1.83746548, 4.83694019, 24.16690909, 49.22484307])
    result = charged_dimensionless_kernel.get_coulomb_kernel_gopalakrishnan2012(
        DIFFUSIVE_KNUDSEN_ARRAY, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_coulomb_chahl2019():
    """Test the coulomb_chahl2019 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # single value
    expected_single = 0.10960430161885967
    result_single = charged_dimensionless_kernel.get_coulomb_kernel_chahl2019(
        DIFFUSIVE_KNUDSEN_SINGLE, COULOMB_POTENTIAL_RATIO_SINGLE
    )
    np.testing.assert_almost_equal(result_single, expected_single, decimal=4)
    # array
    expected = np.array([1.65863442, 4.37444613, 28.05501739, 59.74082667])
    result = charged_dimensionless_kernel.get_coulomb_kernel_chahl2019(
        DIFFUSIVE_KNUDSEN_ARRAY, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_transition_regime_coagulation_edge_cases():
    """Test the transition_regime_coagulation function with edge case values.

    Edge cases tested:
    - Zero values: Should handle zero particle sizes correctly
    - Small values (0.1): Should compute valid coagulation rates
    - Negative values: Should raise ValueError (physically impossible)
    - Very large values: Should compute without numerical overflow
    - NaN: Should raise ValueError (invalid input)
    - Infinity: Should raise ValueError (invalid input)
    """
    # Test zero and small values
    small_zero_input = np.array([0.1, 0.0])
    small_zero_result = charged_dimensionless_kernel.get_hard_sphere_kernel(
        small_zero_input
    )
    small_zero_expected = np.array([0.10960430161885967, 0.0])
    np.testing.assert_almost_equal(
        small_zero_result, small_zero_expected, decimal=4
    )

    # Test very large values (using realistic upper bound for particle sizes)
    large_input = np.array([1e5, 1e5])  # 100mm particles
    large_result = charged_dimensionless_kernel.get_hard_sphere_kernel(
        large_input
    )
    assert np.isfinite(large_result).all(), (
        "Should handle large values without overflow"
    )

    # Test negative values
    negative_input = np.array([-1.0, 1.0])
    with pytest.raises(ValueError, match="Particle sizes must be non-negative"):
        charged_dimensionless_kernel.get_hard_sphere_kernel(negative_input)

    # Test NaN values
    nan_input = np.array([np.nan, 1.0])
    with pytest.raises(ValueError, match="Invalid particle sizes"):
        charged_dimensionless_kernel.get_hard_sphere_kernel(nan_input)

    # Test infinity values
    inf_input = np.array([np.inf, 1.0])
    with pytest.raises(ValueError, match="Invalid particle sizes"):
        charged_dimensionless_kernel.get_hard_sphere_kernel(inf_input)
