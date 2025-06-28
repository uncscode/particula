"""Test for water activity calculation."""

import numpy as np

from particula.activity.water_activity import (
    biphasic_water_activity_point,
    fixed_water_activity,
)


def test_fixed_water_activity():
    """Test for fixed_water_activity function."""
    molar_mass_ratio = 18.016 / 250
    water_activity_desired = np.linspace(0.1, 0.9999, 1000)
    oxygen2carbon = 0.3
    density = 1500

    result = fixed_water_activity(
        water_activity=water_activity_desired,
        molar_mass_ratio=molar_mass_ratio,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )

    assert isinstance(result, tuple)
    assert len(result) == 3


def test_biphasic_water_activity_point():
    """Test for biphasic_water_activity_point function."""
    oxygen2carbon = 0.3
    hydrogen2carbon = 0.3
    molar_mass_ratio = 18 / 250

    # single entry
    activity_point = biphasic_water_activity_point(
        oxygen2carbon=oxygen2carbon,
        hydrogen2carbon=hydrogen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group=None,
    )
    assert np.all(activity_point >= 0)

    # array entry
    oxygen2carbon_array = np.linspace(0, 0.6, 10)
    hydrogen2carbon_array = np.linspace(0, 0.6, 10)
    molar_mass_ratio_array = 18 / np.linspace(200, 500, 10)
    activity_point = biphasic_water_activity_point(
        oxygen2carbon=oxygen2carbon_array,
        hydrogen2carbon=hydrogen2carbon_array,
        molar_mass_ratio=molar_mass_ratio_array,
        functional_group=None,
    )
    assert np.all(activity_point >= 0)

    # edge cases
    activity_point = biphasic_water_activity_point(
        oxygen2carbon=1.0,
        hydrogen2carbon=1.0,
        molar_mass_ratio=1.0,
        functional_group=None,
    )
    assert np.all(activity_point >= 0)
