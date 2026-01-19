"""Tests for the convert_functional_group helper."""

import numpy as np
import pytest
from particula.activity.convert_functional_group import (
    convert_to_oh_equivalent,
)


def test_convert_to_oh_equivalent_no_group():
    """No functional group returns the original ratios."""
    oxygen2carbon = 0.3
    molar_mass_ratio = 18.016 / 250

    new_oxygen2carbon, new_molar_mass_ratio = convert_to_oh_equivalent(
        oxygen2carbon=oxygen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group=None,
    )

    assert new_oxygen2carbon == oxygen2carbon
    assert new_molar_mass_ratio == molar_mass_ratio


def test_convert_to_oh_equivalent_alcohol_matches_legacy():
    """Alcohol adds one oxygen and sixteen mass units."""
    oxygen2carbon = 0.2
    molar_mass_ratio = 0.05

    oc_adjusted, mm_adjusted = convert_to_oh_equivalent(
        oxygen2carbon=oxygen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group="alcohol",
    )

    assert oc_adjusted == oxygen2carbon + 1.0
    assert mm_adjusted == molar_mass_ratio + 16.0


def test_convert_to_oh_equivalent_carboxylic_acid():
    """Carboxylic acids add two oxygens and forty-five mass units."""
    oxygen2carbon = 0.1
    molar_mass_ratio = 0.01

    oc_adjusted, mm_adjusted = convert_to_oh_equivalent(
        oxygen2carbon=oxygen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group="carboxylic_acid",
    )

    assert oc_adjusted == oxygen2carbon + 2.0
    assert mm_adjusted == molar_mass_ratio + 45.0


def test_convert_to_oh_equivalent_ether_handles_arrays():
    """Arrays are adjusted elementwise for ethers."""
    oxygen2carbon = np.array([0.15, 0.25])
    molar_mass_ratio = np.array([0.02, 0.03])

    oc_adjusted, mm_adjusted = convert_to_oh_equivalent(
        oxygen2carbon=oxygen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group="ether",
    )

    assert isinstance(oc_adjusted, np.ndarray)
    assert isinstance(mm_adjusted, np.ndarray)
    assert np.allclose(oc_adjusted, oxygen2carbon + 1.0)
    assert np.allclose(mm_adjusted, molar_mass_ratio + 16.0)


def test_convert_to_oh_equivalent_invalid_group():
    """Unsupported groups raise with the supported list."""
    with pytest.raises(
        ValueError, match='None, "alcohol", "ether", "carboxylic_acid"'
    ):
        convert_to_oh_equivalent(
            oxygen2carbon=0.1,
            molar_mass_ratio=0.01,
            functional_group="ketone",
        )
