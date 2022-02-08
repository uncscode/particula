""" test knudsen number utility

    the knudsen number goes like:
        0   for larger  particles
        inf for smaller particles
"""

import pytest
from particula import u
from particula.utils.particle_ import knudsen_number


def test_kn():
    """ test the knudsen number utility
    """

    radius = 1e-9 * u.m
    mean_free_path_air = 66 * u.nm

    assert (
        knudsen_number(radius, mean_free_path_air) ==
        pytest.approx(66)
    )
    assert (
        knudsen_number(1e-10) ==
        pytest.approx(664)
    )
    assert (
        knudsen_number(1e-3) ==
        pytest.approx(5e-5, rel=1e0)
    )
    assert (
        knudsen_number(1e-20) ==
        pytest.approx(5e12, rel=1e0)
    )
