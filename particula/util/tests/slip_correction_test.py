""" test the slip correction factor calculation
"""

import pytest
from particula import u
from particula.util.slip_correction import slip_correction_factor as slip_correction


def test_slip_correction():
    """ test the slip correction factor calculation

        the slip correction factor is approximately
            ~1      if  radius ~> 1e-6 m  (Kn -> 0)
            ~100    if  radius ~< 1e-9 m
    """

    radius_micron = 1e-6 * u.m
    radius_nano = 1e-9 * u.m
    # mean free path air
    mfp_air = 66.4e-9 * u.m

    assert (
        slip_correction(radius_micron) ==
        pytest.approx(1, rel=1e-1)
    )
    assert (
        slip_correction(radius_nano) ==
        pytest.approx(100, rel=1e0)
    )
    assert (
        slip_correction(radius_micron, mfp_air) ==
        pytest.approx(1, rel=1e-1)
    )
