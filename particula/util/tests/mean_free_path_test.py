""" testing the mean free path calculation
"""

from multiprocessing.sharedctypes import Value
import py
import pytest
from particula import u
from particula.util.mean_free_path import mean_free_path as mfp


def test_mfp():
    """ Testing the mean free path:

        1. test unitless and unit inputs
        2. test correct units
        3. test the calculated value (ref: ~66e-9 m mfp at sdt)
        4. test errors for invalid inputs

    """

    a_mfp = mfp(298 * u.K, 101325 * u.Pa)
    b_mfp = mfp(298, 101325)
    c_mfp = mfp(298 * u.K, 101325 * u.Pa, 0.03 * u.kg / u.mol)

    assert a_mfp == b_mfp
    assert a_mfp.units == u.m
    assert a_mfp.magnitude == pytest.approx(66.4e-9, rel=1e-1)
    assert c_mfp <= a_mfp

    with pytest.raises(ValueError):
        mfp(5 * u.m, 101325 * u.Pa)

    with pytest.raises(ValueError):
        mfp(298 * u.K, 5 * u.m)

    with pytest.raises(ValueError):
        mfp(300 * u.K, 101325 * u.Pa, 0.03 * u.m/u.mol)
