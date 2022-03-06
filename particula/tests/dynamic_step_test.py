""" testing the DynamicStep class
"""

from particula import u
from particula.dynamic_step import DynamicStep


def test_dyns():
    """ test the dynamic step functionalities
    """

    dyns = DynamicStep(
        mode=100,
        nbins=100,
        nparticles=1e5,
        gsigma=1.25,
    )

    assert dyns.coag_kern().u == u.m**3/u.s
    assert dyns.coag_loss().u == u.m**-4/u.s
    assert dyns.coag_gain().u == u.m**-4/u.s
    assert dyns.coag_rate().u == u.m**-4/u.s

    assert dyns.coag_kern().m.shape == (100, 100)
    assert dyns.coag_loss().m.shape == (100,)
    assert dyns.coag_gain().m.shape == (100,)
    assert dyns.coag_rate().m.shape == (100,)
