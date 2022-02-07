""" test the particle mass calc
"""

import numpy as np
from particula import u
from particula.utils import (
    unitless,
)
from particula.utils.particle_ import (
    particle_mass,
)

def test_particle_mass():
    """ testing
    """

    a_radius = 1 * u.m
    a_density = 1000 * u.kg/u.m**3

    b_radius = unitless(a_radius)

    a_mass = particle_mass(a_radius, a_density)
    b_mass = particle_mass(b_radius)

    assert a_mass.magnitude == 1000 * 4 * np.pi / 3
    assert a_mass.units == u.kg
    assert b_mass == a_mass
