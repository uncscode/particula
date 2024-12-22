"""
Gibbs mixing test
"""

import numpy as np
from particula.activity.gibbs_mixing import gibbs_of_mixing
from particula.activity.bat_coefficents import (
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
