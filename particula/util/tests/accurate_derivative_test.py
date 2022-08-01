""" testing the acc4_derivative function
"""

from particula import particle, rates
from particula.util.accurate_derivative import acc4_derivative

p1 = particle.Particle()
r1 = rates.Rates(particle=p1)

def test_derivative():
    """ test derivative """
    assert r1.condensation_growth_speed().m.shape == p1.particle_radius.m.shape
    assert (
        r1.condensation_growth_speed() * p1.particle_radius
    ).m.shape == p1.particle_radius.m.shape
    assert acc4_derivative(
        r1.condensation_growth_speed()*p1.particle_radius,
        p1.particle_radius
    ).m.shape == p1.particle_radius.m.shape
