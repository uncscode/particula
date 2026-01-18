"""Test the ratio module."""

import numpy as np
import pytest

from particula.activity.ratio import from_molar_mass_ratio, to_molar_mass_ratio


def test_to_molar_mass_ratio_scalar():
    """Scalar returns float and matches expected ratio."""
    molar_mass = 30.0
    expected_ratio = 18.01528 / molar_mass
    result = to_molar_mass_ratio(molar_mass)
    assert isinstance(result, float)
    assert result == expected_ratio


def test_to_molar_mass_ratio_array_and_list():
    """List/array inputs return ndarray with expected values."""
    molar_masses = [20.0, 40.0, 60.0]
    expected_ratios = [18.01528 / mm for mm in molar_masses]
    result_list = to_molar_mass_ratio(molar_masses)
    assert isinstance(result_list, np.ndarray)
    assert result_list.shape == (3,)
    assert np.allclose(result_list, expected_ratios)

    molar_masses_array = np.array([72.06, 36.03])
    expected_array = 18.01528 / molar_masses_array
    result_array = to_molar_mass_ratio(molar_masses_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)
    assert np.allclose(result_array, expected_array)


def test_to_molar_mass_ratio_validation():
    """Non-positive molar mass raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        to_molar_mass_ratio(0)
    with pytest.raises(ValueError, match="positive"):
        to_molar_mass_ratio(-1.0)


def test_from_molar_mass_ratio_scalar():
    """Scalar ratio returns float molar mass."""
    molar_mass_ratio = 0.5
    expected_molar_mass = 18.01528 * molar_mass_ratio
    result = from_molar_mass_ratio(molar_mass_ratio)
    assert isinstance(result, float)
    assert result == expected_molar_mass


def test_from_molar_mass_ratio_array_and_list():
    """List/array inputs return ndarray with expected values."""
    molar_mass_ratios = [0.2, 0.4, 0.6]
    expected_molar_masses = [18.01528 * mm for mm in molar_mass_ratios]
    result_list = from_molar_mass_ratio(molar_mass_ratios)
    assert isinstance(result_list, np.ndarray)
    assert result_list.shape == (3,)
    assert np.allclose(result_list, expected_molar_masses)

    molar_mass_ratios_array = np.array([0.25, 1.0])
    expected_array = np.array([4.50382, 18.01528])
    result_array = from_molar_mass_ratio(molar_mass_ratios_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)
    assert np.allclose(result_array, expected_array)


def test_from_molar_mass_ratio_validation():
    """Non-positive ratio raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        from_molar_mass_ratio(0)
    with pytest.raises(ValueError, match="positive"):
        from_molar_mass_ratio(-0.1)
