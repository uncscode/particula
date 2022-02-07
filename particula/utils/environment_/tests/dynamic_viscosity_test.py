""" test the dynamic viscosity routine
"""

import pytest
from particula import u
from particula.utils.environment_ import (
    dynamic_viscosity_air as dyn_vis_air,
)


def test_dynamic_viscosity():

    """ testing the dynamic viscosity:

        1. test unitless and unit inputs
        2. test correct units
        3. test the calculated value
    """

    a_viscosity = dyn_vis_air(298 * u.K)
    b_viscosity = dyn_vis_air(298)

    assert a_viscosity == b_viscosity
    assert a_viscosity.units == u.kg / u.m / u.s
    assert a_viscosity.magnitude == pytest.approx(1.8e-05, rel=1e-1)
