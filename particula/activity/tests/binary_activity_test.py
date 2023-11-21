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


def test_biphasic_water_activity_point():
    """test for biphasic_water_activity_point function."""
    oxygen2carbon = 0.3
    hydrogen2carbon = 0.3
    molar_mass_ratio = 18/250

    # single entry
    activity_point = binary_activity.biphasic_water_activity_point(
        oxygen2carbon=oxygen2carbon,
        hydrogen2carbon=hydrogen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group=None,
    )
    assert np.all(activity_point >= 0)

    # array entry
    oxygen2carbon_array = np.linspace(0, 0.6, 10)
    hydrogen2carbon_array = np.linspace(0, 0.6, 10)
    molar_mass_ratio_array = 18/np.linspace(200, 500, 10)
    activity_point = binary_activity.biphasic_water_activity_point(
        oxygen2carbon=oxygen2carbon_array,
        hydrogen2carbon=hydrogen2carbon_array,
        molar_mass_ratio=molar_mass_ratio_array,
        functional_group=None,
    )
    assert np.all(activity_point >= 0)


def test_gibbs_of_mixing():
    """test for gibbs_of_mixing function."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2.5

    gibbs_mix, dervative_gibbs = binary_activity.gibbs_of_mixing(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        fit_dict=binary_activity.FIT_LOW,
    )
    assert np.all(gibbs_mix >= 0)
    assert np.all(dervative_gibbs ** 2 >= 0)

    # repeat for mid fit
    gibbs_mix, dervative_gibbs = binary_activity.gibbs_of_mixing(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        fit_dict=binary_activity.FIT_MID,
    )
    assert np.all(gibbs_mix >= 0)
    assert np.all(dervative_gibbs ** 2 >= 0)

    # repeat for high fit
    gibbs_mix, dervative_gibbs = binary_activity.gibbs_of_mixing(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        fit_dict=binary_activity.FIT_HIGH,
    )
    assert np.all(gibbs_mix >= 0)
    assert np.all(dervative_gibbs ** 2 >= 0)


def test_activity_coefficents():
    """test for activity_coefficents function."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2.5

    activity_coefficients = binary_activity.activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=None,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)


def test_fixed_water_activity():
    """test for fixed_water_activity function."""
    molar_mass_ratio = 18.016 / 250
    water_activity_desired = np.linspace(0.001, 0.9999, 1000)
    oxygen2carbon = 0.3
    density = 1.5

    _ = binary_activity.fixed_water_activity(
        water_activity=water_activity_desired,
        molar_mass_ratio=molar_mass_ratio,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
