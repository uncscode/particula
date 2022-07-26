""" testing the DynamicStep class
"""

from particula import u
from particula.particle import Particle
from particula.dynamics import Rates


def test_dyns():
    """ test the dynamic step functionalities
    """

    p1 = Particle(
        mode=100,
        nbins=100,
        nparticles=1e5,
        gsigma=1.25,
    )

    dyns = Rates(p1)

    assert dyns.coagulation_loss().u == u.m**-4/u.s
    assert dyns.coagulation_gain().u == u.m**-4/u.s
    assert dyns.coagulation_rate().u == u.m**-4/u.s
    assert dyns.condensation_growth_rate().u == u.m/u.s

    assert dyns.coagulation_loss().m.shape == (100,)
    assert dyns.coagulation_gain().m.shape == (100,)
    assert dyns.coagulation_rate().m.shape == (100,)
    assert dyns.condensation_growth_rate().m.shape == (100,)
