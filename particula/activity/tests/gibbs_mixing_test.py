"""
Gibbs mixing test
"""

import numpy as np
from particula.activity.gibbs_mixing import (
    gibbs_of_mixing,
    gibbs_mix_weight,
)
from particula.activity.bat_coefficients import (
    FIT_HIGH,
    FIT_LOW,
    FIT_MID,
)


def test_gibbs_of_mixing():
    """test for gibbs_of_mixing function."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2500

    gibbs_mix, dervative_gibbs = gibbs_of_mixing(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        fit_dict=FIT_LOW,
    )
    assert np.all(gibbs_mix >= 0)
    assert np.all(dervative_gibbs**2 >= 0)

    # repeat for mid fit
    gibbs_mix, dervative_gibbs = gibbs_of_mixing(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        fit_dict=FIT_MID,
    )
    assert np.all(gibbs_mix >= 0)
    assert np.all(dervative_gibbs**2 >= 0)

    # repeat for high fit
    gibbs_mix, dervative_gibbs = gibbs_of_mixing(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        fit_dict=FIT_HIGH,
    )
    assert np.all(gibbs_mix >= 0)
    assert np.all(dervative_gibbs**2 >= 0)

    # test with array inputs
    molar_mass_ratio_array = np.array([18.016 / 250, 18.016 / 200])
    organic_mole_fraction_array = np.array([0.1, 0.2])
    oxygen2carbon_array = np.array([0.3, 0.4])
    density_array = np.array([2500, 2600])

    gibbs_mix, dervative_gibbs = gibbs_of_mixing(
        molar_mass_ratio=molar_mass_ratio_array,
        organic_mole_fraction=organic_mole_fraction_array,
        oxygen2carbon=oxygen2carbon_array,
        density=density_array,
        fit_dict=FIT_LOW,
    )
    assert np.all(gibbs_mix >= 0)
    assert np.all(dervative_gibbs**2 >= 0)


def test_gibbs_mix_weight():
    """test for gibbs_mix_weight function."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2500

    gibbs_mix, dervative_gibbs = gibbs_mix_weight(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    assert np.all(gibbs_mix >= 0)
    assert np.all(dervative_gibbs**2 >= 0)

    # test with functional group
    gibbs_mix, dervative_gibbs = gibbs_mix_weight(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group="alcohol",
    )
    assert np.all(gibbs_mix >= 0)
    assert np.all(dervative_gibbs**2 >= 0)

    # test with array inputs
    molar_mass_ratio_array = np.array([18.016 / 250, 18.016 / 200])
    organic_mole_fraction_array = np.array([0.1, 0.2])
    oxygen2carbon_array = np.array([0.3, 0.4])
    density_array = np.array([2500, 2600])

    gibbs_mix, dervative_gibbs = gibbs_mix_weight(
        molar_mass_ratio=molar_mass_ratio_array,
        organic_mole_fraction=organic_mole_fraction_array,
        oxygen2carbon=oxygen2carbon_array,
        density=density_array,
    )
    assert np.all(gibbs_mix >= 0)
    assert np.all(dervative_gibbs**2 >= 0)
