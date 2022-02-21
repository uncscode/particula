""" test knudsen number utility

    the knudsen number goes like:
        0   for larger  particles
        inf for smaller particles
"""

import pytest
from particula import u
from particula.util.knudsen_number import knu


def test_kn():
    """ test the knudsen number utility
    """

    radius = 1e-9 * u.m
    mfp_air = 66 * u.nm

    assert (
        knu(radius=radius, mfp=mfp_air) ==
        pytest.approx(66)
    )

    assert (
        knu(radius=1e-10) ==
        pytest.approx(664, rel=1e-1)
    )

    assert (
        knu(radius=1e-3) ==
        pytest.approx(5e-5, rel=1e0)
    )

    assert (
        knu(radius=1e-20) ==
        pytest.approx(5e12, rel=1e0)
    )

    assert knu(radius=[1, 2, 3]).m.shape == (3,)
    assert knu(radius=1, mfp=[1, 2, 3]).m.shape == (3, 1)
    assert knu(radius=[1, 2, 3], mfp=[1, 2, 3]).m.shape == (3, 3)
    assert knu(radius=[1, 2, 3], temperature=[1, 2, 3]).m.shape == (3, 3)
    assert knu(radius=[1, 2, 3], pressure=[1, 2, 3]).m.shape == (3, 3)
    assert knu(radius=[1, 2, 3], molecular_weight=[1, 2, 3]).m.shape == (3, 3)

    assert knu(
        radius=[1, 2, 3], temperature=[1, 2, 3], pressure=[1, 2, 3]
    ).m.shape == (3, 3)
