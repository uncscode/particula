"""
Test for activity_coefficents function.
"""

import numpy as np
from particula.activity.activity_coefficients import bat_activity_coefficients


def test_activity_coefficents():
    """test for activity_coefficents function."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2500

    activity_coefficients = bat_activity_coefficients(
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
