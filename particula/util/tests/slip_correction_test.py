""" test the slip correction factor calculation
"""

import pytest
from particula import u
from particula.util.knudsen_number import knu
from particula.util.slip_correction import scf


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
    knu_val = knu(radius=radius_micron, mfp=mfp_air)

    assert (
        scf(radius=radius_micron) ==
        pytest.approx(1, rel=1e-1)
    )

    assert (
        scf(radius=radius_nano) ==
        pytest.approx(100, rel=1e0)
    )

    assert (
        scf(radius=radius_micron, knu=knu_val) ==
        pytest.approx(1, rel=1e-1)
    )

    assert scf(radius=[1, 2, 3]).m.shape == (3,)
    assert scf(radius=1, mfp=[1, 2, 3]).m.shape == (3, 1)
    assert scf(radius=[1, 2, 3], mfp=[1, 2, 3]).m.shape == (3, 3)
    assert scf(radius=[1, 2, 3], temperature=[1, 2, 3]).m.shape == (3, 3)
