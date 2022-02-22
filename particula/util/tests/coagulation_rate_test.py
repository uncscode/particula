""" testing the coagulation rate
"""

from particula import u
from particula.util.coagulation_rate import CoagulationRate

CoagRate = CoagulationRate(
    mode=100,
    nbins=1000,
    nparticles=1e5,
    gsigma=1.25,
)

rads = CoagRate.rad()

lnds = CoagRate.lnd()

kern = CoagRate.coag_kern()

loss = CoagRate.coag_loss()


def test_kern():
    """ first test the kernel
    """

    assert kern.u == u.m**3/u.s
    assert kern.m.shape == rads.shape + rads.shape


def test_loss():
    """ test the loss
    """

    assert loss.u == u.m**-3/u.s
    assert loss.m.shape == rads.shape
    assert loss.m.shape == lnds.shape

