""" testing the coagulation rate
"""
import numpy as np
import pytest
from particula import u
from particula.aerosol_dynamics.particle_distribution import \
    ParticleDistribution
from particula.util.coagulation_rate import CoagulationRate
from particula.util.dimensionless_coagulation import full_coag

distribution = ParticleDistribution(
    cutoff=.9999,
    mode=100e-9,
    nbins=1000,
    nparticles=1e5,
    gsigma=1.25,
).distribution()

radius = ParticleDistribution(
    cutoff=.9999,
    mode=100e-9,
    nbins=1000,
    nparticles=1e5,
    gsigma=1.25,
).radius()

kernel = full_coag(
    radius=radius,
    mode=100e-9,
    nbins=1000,
    nparticles=1e5,
    gsigma=1.25,
)

CoagRate = CoagulationRate(
    distribution=distribution,
    radius=radius,
    kernel=kernel,
)

rads = radius

lnds = distribution

kern = kernel

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

    assert loss.u == u.m**-4/u.s
    assert loss.m.shape == rads.shape
    assert loss.m.shape == lnds.shape


def test_gain():
    """ test the gain
    """

    assert gain.size == rads.size
    assert gain.u == u.m**-4/u.s


def test_mass():
    """ test mass conservation
    """

    assert np.trapz((gain - loss)*rads**3, rads).u == u.s**-1
    assert np.trapz((gain - loss)*rads**3, rads) <= 0.2 * u.s**-1
    assert np.trapz(
        (gain - loss).to(u.cm**-4/u.s)*rads**3,rads
    ).m == pytest.approx(0.0, abs=1e-8)
