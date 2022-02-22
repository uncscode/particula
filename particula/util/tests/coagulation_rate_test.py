""" testing the coagulation rate
"""
import numpy as np
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

gain = CoagRate.coag_gain()


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


def test_gain():
    """ test the gain
    """

    assert gain.size == rads.size
    assert gain.u == u.m**-3/u.s


def test_mass():
    """ test mass conservation
    """

    assert np.trapz((gain - loss)*rads**2*u.m**2, rads*u.m).u == u.s**-1
