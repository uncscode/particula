""" testing the cbar calc
"""

import pytest
from particula import u
from particula.util.rms_speed import cbar


def test_cbar():
    """ Testing the mean speed of molecules:

        1. test unitless and unit inputs
        2. test correct units
        3. test errors for invalid inputs

    """

    a_cbar = cbar()
    b_cbar = cbar(temperature=298.15)
    c_cbar = cbar(temperature=300, molecular_weight=0.03)

    assert a_cbar == b_cbar
    assert a_cbar.units == u.m/u.s
    assert c_cbar <= a_cbar

    assert cbar(temperature=[200, 300]).m.shape == (2,)
    assert cbar(molecular_weight=[0.03, 0.04]).m.shape == (2,)
    assert cbar(
        temperature=[200, 300], molecular_weight=[0.03, 0.04]
    ).m.shape == (2,)

    with pytest.raises(ValueError):
        cbar(temperature=5*u.m)

    with pytest.raises(ValueError):
        cbar(temperature=298*u.K, molecular_weight=5*u.m)
