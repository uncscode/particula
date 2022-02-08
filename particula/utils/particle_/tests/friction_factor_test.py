""" test friction factor calculation
"""

import pytest
from particula import u
from particula.utils.particle_ import friction_factor


def test_friction_factor():
    """ test the friction factor calculation

        In the continuum limit (Kn -> 0; Cc -> 1):
            6 * np.pi * dyn_vis_air * radius

        In the kinetic limit (Kn -> inf):
            8.39 * (dyn_vis_air/mfp_air) * const * radius**2

        See more: DOI: 10.1080/02786826.2012.690543 (const=1.36)
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

    assert (
        friction_factor(1e-5).magnitude ==
        pytest.approx(6*3.14*1e-5 * dyn_vis_air.magnitude, rel=1e-1)
    )
    assert (
        friction_factor(1e-10).magnitude ==
        pytest.approx(8.39*1.36e-20 * (dyn_vis_air/mfp_air).magnitude)
    )
