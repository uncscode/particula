"""Test for transition_regime calculation functions."""

import numpy as np

from particula.dynamics.coagulation import transition_regime

# test single value
DIFFUSIVE_KNUDSEN_SINGLE = 0.1
COULOMB_POTENTIAL_RATIO_SINGLE = 0.5

# test array
DIFFUSIVE_KNUDSEN_ARRAY = np.array([0.5, 1.0, 5.0, 10.0])
COULOMB_POTENTIAL_RATIO_ARRAY = np.array([0.7, 0.9, 1.0, 1.1])


def test_hard_sphere():
    """
    Test the hard_sphere function with a single value and
    array of diffusive_knudsen values.
    """
    # single value
    expected_single = 0.10960430161885967
    result_single = transition_regime.hard_sphere(DIFFUSIVE_KNUDSEN_SINGLE)
    np.testing.assert_almost_equal(result_single, expected_single, decimal=4)
    # array
    expected = np.array([1.65971644, 4.12694075, 24.16690909, 49.22484307])
    result = transition_regime.hard_sphere(DIFFUSIVE_KNUDSEN_ARRAY)
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_coulomb_dyachkov2007():
    """
    Test the coulomb_dyachkov2007 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # single value
    expected_single = 0.10960430161885967
    result_single = transition_regime.coulomb_dyachkov2007(
        DIFFUSIVE_KNUDSEN_SINGLE, COULOMB_POTENTIAL_RATIO_SINGLE
    )
    np.testing.assert_almost_equal(result_single, expected_single, decimal=4)
    # array
    expected = np.array([1.73703563, 4.60921277, 26.22159795, 51.92102133])
    result = transition_regime.coulomb_dyachkov2007(
        DIFFUSIVE_KNUDSEN_ARRAY, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_coulomb_gatti2008():
    """
    Test the coulomb_gatti2008 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # single value
    expected_single = 0.12621027
    result_single = transition_regime.coulomb_gatti2008(
        DIFFUSIVE_KNUDSEN_SINGLE, COULOMB_POTENTIAL_RATIO_SINGLE
    )
    np.testing.assert_almost_equal(result_single, expected_single, decimal=4)
    # array
    expected = np.array([2.00132915, 5.10865767, 26.42422258, 52.43789491])
    result = transition_regime.coulomb_gatti2008(
        DIFFUSIVE_KNUDSEN_ARRAY, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_coulomb_gopalakrishnan2012():
    """
    Test the coulomb_gopalakrishnan2012 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # single value
    expected_single = 0.1096043
    result_single = transition_regime.coulomb_gopalakrishnan2012(
        DIFFUSIVE_KNUDSEN_SINGLE, COULOMB_POTENTIAL_RATIO_SINGLE
    )
    np.testing.assert_almost_equal(result_single, expected_single, decimal=4)
    # array
    expected = np.array([1.83746548, 4.83694019, 24.16690909, 49.22484307])
    result = transition_regime.coulomb_gopalakrishnan2012(
        DIFFUSIVE_KNUDSEN_ARRAY, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    np.testing.assert_almost_equal(result, expected, decimal=4)


def test_coulomb_chahl2019():
    """
    Test the coulomb_chahl2019 function with a single value and
    array of diffusive_knudsen and coulomb_potential_ratio values.
    """
    # single value
    expected_single = 0.10960430161885967
    result_single = transition_regime.coulomb_chahl2019(
        DIFFUSIVE_KNUDSEN_SINGLE, COULOMB_POTENTIAL_RATIO_SINGLE
    )
    np.testing.assert_almost_equal(result_single, expected_single, decimal=4)
    # array
    expected = np.array([1.65863442, 4.37444613, 28.05501739, 59.74082667])
    result = transition_regime.coulomb_chahl2019(
        DIFFUSIVE_KNUDSEN_ARRAY, COULOMB_POTENTIAL_RATIO_ARRAY
    )
    np.testing.assert_almost_equal(result, expected, decimal=4)
