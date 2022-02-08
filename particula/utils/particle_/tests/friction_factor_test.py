""" test friction factor calculation
"""

import pytest
from particula import u
from particula.utils.particle_ import friction_factor


def test_friction_factor():
    """ test the friction factor calculation
    """

    radius = 1e-9 * u.m
    dyn_vis_air = 1.8e-05 * u.N * u.s / u.m
    mfp_air = 66.4e-9 * u.m

    assert (
        friction_factor(radius, dyn_vis_air, mfp_air) ==
        pytest.approx(3e-15)
    )
    assert (
        friction_factor(radius, dyn_vis_air) ==
        pytest.approx(3e-15)
    )
    assert (
        friction_factor(radius) ==
        pytest.approx(3e-15)
    )
    assert (
        friction_factor(radius, dyn_vis_air, mfp_air).units ==
        (1 * u.N * u.s).to_base_units()
    )
