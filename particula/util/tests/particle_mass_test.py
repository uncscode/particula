""" test the particle mass calculation
"""

import numpy as np
import pytest
from particula import u
from particula.util.particle_mass import mass


def test_particle_mass():
    """ Testing the particle mass calculation:

            * check defaults
            * check raise errors
            * check output units/mag
            * ensures radius is mandatory
    """

    a_radius = 1 * u.m
    a_density = 1000 * u.kg/u.m**3

    b_radius = a_radius.m

    a_mass = mass(radius=a_radius, density=a_density)
    b_mass = mass(radius=b_radius)

    assert a_mass.magnitude == 1000 * 4 * np.pi / 3
    assert a_mass.units == u.kg
    assert b_mass == a_mass

    assert mass(radius=[1e-9, 1e-8]).m.shape == (2,)
    assert mass(radius=1-9, density=[1e3, 1e4]).m.shape == (2,)
    assert mass(radius=[1e-9, 1e-8], density=[1e3, 1e4]).m.shape == (2,)

    with pytest.raises(ValueError):
        mass(radius=1*u.kg)

    with pytest.raises(ValueError):
        mass(radius=1*u.m, density=1*u.kg)

    with pytest.raises(ValueError):
        mass(radius=1*u.m, shape_factor=1*u.m)

    with pytest.raises(ValueError):
        mass(radius=1*u.m, volume_void=1*u.kg)

    with pytest.raises(ValueError):
        mass()
