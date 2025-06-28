"""Test for convert_functional_group.py."""

from particula.activity.convert_functional_group import (
    convert_to_oh_equivalent,
)


def test_convert_to_oh_eqivalent():
    """Test for convert_to_oh_equivalent function."""
    molar_mass_ratio = 18.016 / 250
    oxygen2carbon = 0.3

    new_oxygen2carbon, new_molar_mass_ratio = convert_to_oh_equivalent(
        oxygen2carbon=oxygen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group=None,
    )
    assert new_oxygen2carbon == oxygen2carbon
    assert new_molar_mass_ratio == molar_mass_ratio
