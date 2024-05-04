""" test the slip correction factor calculation
"""

import numpy as np
import pytest
from particula import u
from particula.util.knudsen_number import knu
from particula.util.slip_correction import scf, cunningham_slip_correction


def test_slip_correction():
    """ test the slip correction factor calculation

        the slip correction factor is approximately
            ~1      if  radius ~> 1e-6 m  (Kn -> 0)
            ~100    if  radius ~< 1e-9 m
    """

    radius_micron = 1e-6 * u.m
    radius_nano = 1e-9 * u.m
    # mean free path air
    mfp_air = 66.4e-9 * u.m
    knu_val = knu(radius=radius_micron, mfp=mfp_air)

    assert (
        scf(radius=radius_micron) ==
        pytest.approx(1, rel=1e-1)
    )

    assert (
        scf(radius=radius_nano) ==
        pytest.approx(100, rel=1e0)
    )

    assert (
        scf(radius=radius_micron, knu=knu_val) ==
        pytest.approx(1, rel=1e-1)
    )

    assert scf(radius=[1, 2, 3]).m.shape == (3,)
    assert scf(radius=1, mfp=[1, 2, 3]).m.shape == (3, 1)
    assert scf(radius=[1, 2, 3], mfp=[1, 2, 3]).m.shape == (3, 3)
    assert scf(radius=[1, 2, 3], temperature=[1, 2, 3]).m.shape == (3, 3)


def test_cunningham_slip_correction_basic():
    """Test with basis input values"""
    knudsen_number = 0.5
    expected = 1 + knudsen_number * \
        (1.257 + 0.4 * np.exp(-1.1 / knudsen_number))
    assert np.isclose(cunningham_slip_correction(knudsen_number), expected)


def test_cunningham_slip_correction_array():
    """Test with an array of Knudsen numbers"""
    knudsen_numbers = np.array([0.1, 1, 10])
    expected = 1 + knudsen_numbers * \
        (1.257 + 0.4 * np.exp(-1.1 / knudsen_numbers))
    np.testing.assert_allclose(
        cunningham_slip_correction(knudsen_numbers), expected)


def test_cunningham_slip_correction_high_value():
    """Test with a high Knudsen number to check behavior approaching"""
    knudsen_number = 100
    expected = 1 + knudsen_number * \
        (1.257 + 0.4 * np.exp(-1.1 / knudsen_number))
    assert np.isclose(cunningham_slip_correction(knudsen_number), expected)


def test_cunningham_slip_correction_low_value():
    """Test with a low Knudsen number (approaching zero)"""
    knudsen_number = 0.01
    expected = 1 + knudsen_number * \
        (1.257 + 0.4 * np.exp(-1.1 / knudsen_number))
    assert np.isclose(cunningham_slip_correction(knudsen_number), expected)


def test_cunningham_slip_correction_negative_value():
    """Test with a negative Knudsen number to see if the function handles it,
    remove once value error handling is implemented in the function"""
    knudsen_number = -0.5
    expected = 1 + knudsen_number * \
        (1.257 + 0.4 * np.exp(-1.1 / knudsen_number))
    assert np.isclose(cunningham_slip_correction(knudsen_number), expected)


def test_cunningham_slip_correction_type_error():
    """Test with an incorrect type for the Knudsen number input"""
    knudsen_number = "0.5"  # Incorrect type (string)
    with pytest.raises(TypeError):
        cunningham_slip_correction(knudsen_number)
