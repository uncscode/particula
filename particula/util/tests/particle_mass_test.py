""" test the particle mass calculation
"""

import pytest
import numpy as np
from particula import u
from particula.util.particle_mass import particle_mass


def test_particle_mass():
    """ Testing the particle mass calculation:

            * check defaults
            * check raise errors
            * check output units/mag
    """

    a_radius = 1 * u.m
    a_density = 1000 * u.kg/u.m**3

    b_radius = a_radius.m

    a_mass = particle_mass(a_radius, a_density)
    b_mass = particle_mass(b_radius)

    assert a_mass.magnitude == 1000 * 4 * np.pi / 3
    assert a_mass.units == u.kg
    assert b_mass == a_mass

    with pytest.raises(ValueError):
        particle_mass(1*u.kg)

    with pytest.raises(ValueError):
        particle_mass(1*u.m, 1*u.kg)
