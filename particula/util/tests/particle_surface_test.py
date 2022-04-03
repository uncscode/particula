""" test the particle mass calculation
"""

import numpy as np
import pytest
from particula import u
from particula.util.particle_surface import area


def test_particle_mass():
    """ Testing the particle area calculation:

            * check defaults
            * check raise errors
            * check output units/mag
            * ensures radius is mandatory
    """

    a_radius = 1 * u.m

    b_radius = a_radius.m

    a_area = area(radius=a_radius)
    b_area = area(radius=b_radius)

    assert a_area.magnitude == 4 * np.pi
    assert a_area.units == u.m**2
    assert b_area == a_area

    assert area(radius=[1e-9, 1e-8]).m.shape == (2,)
    assert area(radius=1-9, area_factor=[1, 0.9]).m.shape == (2,)
    assert area(radius=[1e-9, 1e-8], area_factor=[1, 0.9]).m.shape == (2,)

    with pytest.raises(ValueError):
        area(radius=1*u.kg)

    with pytest.raises(ValueError):
        area(radius=1*u.m, area_factor=1*u.kg)

    with pytest.raises(ValueError):
        area()
