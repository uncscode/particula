"""Test the ratio module."""
from particula.activity.ratio import to_molar_mass_ratio, from_molar_mass_ratio


def test_to_molar_mass_ratio():
    """Test the to_molar_mass_ratio function."""
    # Test case 1: Single molar mass
    molar_mass = 30.0
    expected_ratio = 18.01528 / molar_mass
    assert to_molar_mass_ratio(molar_mass) == expected_ratio

    # Test case 2: List of molar masses
    molar_masses = [20.0, 40.0, 60.0]
    expected_ratios = [18.01528 / mm for mm in molar_masses]
    assert to_molar_mass_ratio(molar_masses) == expected_ratios


def test_from_molar_mass_ratio():
    """Test the from_molar_mass_ratio function."""
    # Test case 1: Single molar mass ratio
    molar_mass_ratio = 0.5
    expected_molar_mass = 18.01528 * molar_mass_ratio
    assert from_molar_mass_ratio(molar_mass_ratio) == expected_molar_mass

    # Test case 2: List of molar mass ratios
    molar_mass_ratios = [0.2, 0.4, 0.6]
    expected_molar_masses = [18.01528 * mm for mm in molar_mass_ratios]
    assert from_molar_mass_ratio(molar_mass_ratios) == expected_molar_masses
