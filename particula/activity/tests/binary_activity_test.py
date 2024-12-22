"""Tests if the binary activity is working, does not
verify values, just that it can be called."""

import numpy as np

from particula.activity import binary_activity


def test_convert_to_oh_eqivalent():
    """test for convert_to_oh_equivalent function."""

    molar_mass_ratio = 18.016 / 250
    oxygen2carbon = 0.3

    new_oxygen2carbon, new_molar_mass_ratio = \
        binary_activity.convert_to_oh_equivalent(
            oxygen2carbon=oxygen2carbon,
            molar_mass_ratio=molar_mass_ratio,
            functional_group=None
        )
    assert new_oxygen2carbon == oxygen2carbon
    assert new_molar_mass_ratio == molar_mass_ratio


def test_bat_blending_weights():
    """test for bat_blending_weights function."""
    molar_mass_ratio = 18.016 / 250
    oxygen2carbon = 0.3

    weights = binary_activity.bat_blending_weights(
        molar_mass_ratio=molar_mass_ratio,
        oxygen2carbon=oxygen2carbon
    )
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)







