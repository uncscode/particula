""" test friction factor calculation
"""

import pytest
from particula import u
from particula.util.friction_factor import frifac


def test_friction_factor():
    """ test the friction factor calculation

        In the continuum limit (Kn -> 0; Cc -> 1):
            6 * np.pi * dyn_vis_air * radius

        In the kinetic limit (Kn -> inf):
            8.39 * (dyn_vis_air/mfp_air) * const * radius**2

        See more: DOI: 10.1080/02786826.2012.690543 (const=1.36)
    """

    rad = 1e-9 * u.m
    dva = 1.8e-05 * u.N * u.s / u.m**2
    mfp = 65e-9 * u.m

    assert (
        frifac(radius=rad, dynamic_viscosity=dva) ==
        pytest.approx(3e-15)
    )

    assert (
        frifac(radius=rad) ==
        pytest.approx(3e-15)
    )
    assert (
        frifac(radius=rad).units ==
        (1 * u.N * u.s / u.m).to_base_units()
    )

    assert (
        frifac(radius=1e-5).magnitude ==
        pytest.approx(6*3.14*1e-5 * dva.magnitude, rel=1e-1)
    )
    assert (
        frifac(radius=1e-10).magnitude ==
        pytest.approx(8.39*1.36e-20 * (dva/mfp).magnitude)
    )

    assert frifac(radius=[1, 2, 3]).m.shape == (3,)
    assert frifac(radius=[1, 2, 3], temperature=[1, 2, 3]).m.shape == (3, 3)
    assert frifac(radius=[1, 2, 3], pressure=[1, 2, 3]).m.shape == (3, 3)
