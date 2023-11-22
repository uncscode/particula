"""Test the convert module."""

import numpy as np
import pytest
from particula.util import convert


def test_coerce_type():
    """Test the coerce_type function."""
    # Test int input
    assert convert.coerce_type(42, int) == 42

    # Test float input
    assert convert.coerce_type(3.14, float) == 3.14

    # Test string input
    assert convert.coerce_type("hello", str) == "hello"

    # Test list input
    assert np.array_equal(
            convert.coerce_type([1, 2, 3], np.ndarray),
            np.array([1, 2, 3])
        )

    # Test tuple input
    assert np.array_equal(
            convert.coerce_type((1, 2, 3), np.ndarray),
            np.array([1, 2, 3])
        )

    # Test array input
    assert np.array_equal(
            convert.coerce_type(np.array([1, 2, 3]), np.ndarray),
            np.array([1, 2, 3])
        )

    # Test invalid conversion
    try:
        convert.coerce_type("hello", int)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


def test_round_arbitrary():
    """Test the round function."""
    # Test single float value
    assert convert.round_arbitrary(3.14, base=0.5, mode='round') == 3.0

    # Test list of float values
    assert np.array_equiv(
            convert.round_arbitrary([1.2, 2.5, 3.8], base=1.0, mode='round'),
            np.array([1.0, 2.0, 4.0])
        )

    # Test NumPy array of float values
    assert np.array_equal(convert.round_arbitrary(
            np.array([1.2, 2.5, 3.8]), base=1.0, mode='floor'),
            np.array([1.0, 2.0, 3.0]))

    # Test NumPy array of ceil values
    assert np.array_equal(convert.round_arbitrary(
            np.array([1.2, 2.5, 3.8]), base=1.0, mode='ceil'),
            np.array([2.0, 3.0, 4.0]))

    # Test rounding to non-integer base
    assert convert.round_arbitrary(3.14, base=0.1, mode='round') == 3.1

    # Test rounding mode "round_nonzero"
    assert np.array_equal(convert.round_arbitrary(
            [0, 0.2, 0.3, 0.6, 3], base=1.0, mode='round', nonzero_edge=True),
            np.array([0, 0.2, 0.3, 1, 3]))

    # Test invalid mode parameter
    try:
        convert.round_arbitrary([1.2, 2.5, 3.8], base=1.0, mode='invalid')
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


def test_radius_diameter():
    """Test the radius_diameter function."""
    # Test radius to diameter conversion
    assert convert.radius_diameter(1.0, to_diameter=True) == 2.0
    assert convert.radius_diameter(2.5, to_diameter=True) == 5.0
    assert convert.radius_diameter(3.8, to_diameter=True) == 7.6

    # Test diameter to radius conversion
    assert convert.radius_diameter(2.0, to_diameter=False) == 1.0
    assert convert.radius_diameter(5.0, to_diameter=False) == 2.5
    assert convert.radius_diameter(7.6, to_diameter=False) == 3.8


def test_volume_to_length():
    """Test the volume_to_length function."""
    # Test radius conversion
    assert np.isclose(convert.volume_to_length(1000), 6.2035, rtol=1e-4)

    # Test diameter conversion
    assert np.isclose(
            convert.volume_to_length(1000, length_type='diameter'),
            12.4071, rtol=1e-4
        )

    # Test incorrect length type
    with pytest.raises(ValueError):
        convert.volume_to_length(1000, length_type='invalid')


def test_length_to_volume():
    """Test the length_to_volume function."""
    # Test with radius
    assert np.isclose(
            convert.length_to_volume(3),
            113.0973,
            rtol=1e-4
        )

    # Test with diameter
    assert np.isclose(
            convert.length_to_volume(6, 'diameter'),
            113.09733552923254,
            rtol=1e-4
        )

    # Test with invalid length_type
    try:
        convert.length_to_volume(3, 'invalid')
    except ValueError as custom_error:
        assert str(custom_error) == 'length_type must be radius or diameter'


def test_kappa_volume_solute():
    """Test the kappa_volume_solute function."""
    # Test with water_activity = 0.95 and kappa = 0.4
    assert np.allclose(
                convert.kappa_volume_solute(100, 0.4, 0.95),
                11.6279,
                rtol=1e-4
            )

    # Test with water_activity = 1 and kappa = 0.0 (zero kappa correction)
    assert np.allclose(
                convert.kappa_volume_solute(200, 0.0, 1),
                0.0,
                rtol=1e-4
            )


def test_kappa_volume_water():
    """Test the kappa_volume_water function."""
    # Test with water_activity = 0.95 and kappa = 0.4
    assert np.allclose(
                convert.kappa_volume_water(5, 0.5, 0.99),
                247.49,
                rtol=1e-4
            )

    # Test with water_activity = 1 and kappa = 0.0 (zero correction)
    assert np.allclose(
                convert.kappa_volume_solute(5, 0.5, 1),
                1.1258e16,
                rtol=1e14
            )


def test_kappa_from_volume():
    """Test the kappa_from_volume function."""
    # Test with water_activity = 0.95 and kappa = 0.4
    assert np.allclose(
                convert.kappa_from_volume(100, 200, 0.5),
                2.0,
                rtol=1e-2
            )

    # Test with water_activity = 1 (zero kappa correction)
    assert np.allclose(
                convert.kappa_from_volume(100, 200, 1),
                4.44089e-16,
                rtol=1e-17
            )


def test_mole_fraction_to_mass_fraction():
    """Test the mole_fraction_to_mass_fraction function."""

    # Test with mole_fraction0 = 0.5, molecular_weight0 = 50,
    assert convert.mole_fraction_to_mass_fraction(
        mole_fraction0=0.5,
        molecular_weight0=50,
        molecular_weight1=100
        ) == (0.3333333333333333, 0.6666666666666667)


def test_mole_fraction_to_mass_fraction_multi():
    """Test the mole_fraction_to_mass_fraction_multi function."""

    # Test with mole_fractions = [0.25, 0.25, 0.5],
    # molecular_weights = [200, 200, 100]
    assert np.allclose(
            convert.mole_fraction_to_mass_fraction_multi(
                mole_fractions=[0.25, 0.25, 0.5],
                molecular_weights=[200, 200, 100]
                ),
            np.array(
                [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
                ),
            rtol=1e-4
        )


def test_mass_fraction_to_volume_fraction():
    """Test the mass_fraction_to_volume_fraction function."""

    # Test with mass_fraction0 = 0.5, density0 = 50,
    assert convert.mass_fraction_to_volume_fraction(
        0.5,
        1.5,
        2
        ) == (0.5714285714285715, 0.4285714285714285)


def test_volume_water_from_volume_fraction():
    """Test the volume_water_from_volume_fraction function."""

    # Test with volume_fraction = 0.5, density = 50,
    assert convert.volume_water_from_volume_fraction(
            volume_solute_dry=100,
            volume_fraction_water=0.8
        ) == 400.0000000000001


def test_effective_refractive_index():
    """Test the effective_refractive_index function."""

    assert convert.effective_refractive_index(
            m_zero=1.5+0.5j,
            m_one=1.33,
            volume_zero=10,
            volume_one=5,
        ) == (1.4572585227821824+0.3214931829339477j)
