""" testing the mean free path calculation
"""

import pytest
from particula import u
from particula.util.mean_free_path import mean_free_path_air as mfp_air


def test_mfp():
    """ Testing the mean free path:

        1. test unitless and unit inputs
        2. test correct units
        3. test the calculated value (ref: ~66e-9 m mfp at sdt)
        4. test errors for invalid inputs

    """

    a_mfp = mfp_air(298 * u.K, 101325 * u.Pa)
    b_mfp = mfp_air(298, 101325)

    assert a_mfp == b_mfp
    assert a_mfp.units == u.m
    assert a_mfp.magnitude == pytest.approx(66.4e-9, rel=1e-1)

    with pytest.raises(ValueError):
        mfp_air(5 * u.m, 101325 * u.Pa)

    with pytest.raises(ValueError):
        mfp_air(298 * u.K, 5 * u.m)
