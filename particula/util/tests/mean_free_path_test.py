""" testing the mean free path calculation
"""

import pytest
from particula import u
from particula.util.mean_free_path import mfp


def test_mfp():
    """ Testing the mean free path:

        1. test unitless and unit inputs
        2. test correct units
        3. test the calculated value (ref: ~66e-9 m mfp at sdt)
        4. test errors for invalid inputs

    """

    a_mfp = mfp(temperature=298*u.K, pressure=101325 * u.Pa)
    b_mfp = mfp(temperature=298, pressure=101325)
    c_mfp = mfp(temperature=298, pressure=101325, molecular_weight=0.03)

    assert a_mfp == b_mfp
    assert a_mfp.units == u.m
    assert a_mfp.magnitude == pytest.approx(66.4e-9, rel=1e-1)
    assert c_mfp <= a_mfp

    assert mfp(temperature=[200, 300]).m.shape == (2,)
    assert mfp(pressure=[1e5, 1.1e5]).m.shape == (2,)
    assert mfp(temperature=[200, 300], pressure=[1e5, 1.1e5]).m.shape == (2,)

    with pytest.raises(ValueError):
        mfp(temperature=5*u.m, pressure=101325*u.Pa)

    with pytest.raises(ValueError):
        mfp(temperature=298*u.K, pressure=5*u.m)

    with pytest.raises(ValueError):
        mfp(temperature=300*u.K,
            pressure=101325*u.Pa,
            molecular_weight=0.03*u.m/u.mol,
            )
