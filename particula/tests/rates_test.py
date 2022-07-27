""" testing the DynamicStep class
"""

from particula import u
from particula.particle import Particle
from particula.rates import Rates


def test_dyns():
    """ test the dynamic step functionalities
    """

    some_particle = Particle(
        mode=100e-9,
        nbins=100,
        nparticles=1e5,
        gsigma=1.25,
        particle_formation_rate=0.5e8,
    )

    dyns = Rates(some_particle)

    assert dyns.coagulation_loss().u == u.m**-4/u.s
    assert dyns.coagulation_gain().u == u.m**-4/u.s
    assert dyns.coagulation_rate().u == u.m**-4/u.s
    assert dyns.condensation_growth_speed().u == u.m/u.s
    assert dyns.condensation_growth_rate().u == u.m**-4/u.s
    assert dyns.nucleation_rate().u == u.m**-4/u.s

    assert dyns.coagulation_loss().m.shape == (100,)
    assert dyns.coagulation_gain().m.shape == (100,)
    assert dyns.coagulation_rate().m.shape == (100,)
    assert dyns.condensation_growth_speed().m.shape == (100,)
    assert dyns.condensation_growth_rate().m.shape == (100,)
    assert dyns.nucleation_rate().m.shape == (100,)

    assert dyns.nucleation_rate().m[0] == 0.5e8
    assert dyns.nucleation_rate().m[1:].max() == 0
