""" test the mean free path calc
"""

import pytest
from particula import u
from particula.utils.environment_ import (
    mean_free_path_air as mfp_air,
)


def test_mfp():

    """ testing the mean free path:

        1. test unitless and unit inputs
        2. test correct units
        3. test the calculated value (ref: ~66 nm mfp at sdt)
    """

    a_mfp = mfp_air(298 * u.K, 101325 * u.Pa)
    b_mfp = mfp_air(298, 101325)

    assert a_mfp == b_mfp
    assert a_mfp.units == u.m
    assert a_mfp.magnitude == pytest.approx(66.4e-9, rel=1e-1)
