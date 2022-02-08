""" test knudsen number utility
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
