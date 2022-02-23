""" test the particle distribution class
"""
import numpy as np
import pytest
from scipy.stats import lognorm

from particula import u
from particula.aerosol_dynamics.particle_distribution import \
    ParticleDistribution


def test_rad():
    """ testing the radius properties
    """

    pdist = ParticleDistribution(
        cutoff=.9999,
        mode=100,
        nbins=1000,
        nparticles=1e5,
        gsigma=1.25,
        volume=1e-6,
    )

    assert 100 >= pdist.radius()[0].m
    assert 100 <= pdist.radius()[-1].m
    assert 1000 == len(pdist.radius())


def test_discretize():
    """ testing the discretization
    """

    pdist = ParticleDistribution(
        mode=100,
        nbins=100,
        nparticles=1e5,
        gsigma=1.25,
    )

    assert 0 <= pdist.discretize()[0]
    assert 0 <= pdist.discretize()[-1]
    assert np.trapz(pdist.discretize(), pdist.radius(
    ).m) == pytest.approx(.9999, rel=1e-1)
    assert len(pdist.discretize()) == len(pdist.radius().m)
    assert lognorm.fit(pdist.discretize())[0] <= 3.75
    assert lognorm.fit(pdist.discretize())[1] <= 1e-5


def test_dist():
    """ test distribution properties
    """

    pdist = ParticleDistribution(
        mode=100,
        nbins=1000,
        nparticles=1e5,
        gsigma=1.25,
    )

    dps = pdist.radius().m
    wts = pdist.discretize()/np.sum(pdist.discretize())

    samples = np.random.choice(dps, size=len(dps), p=wts)

    assert (
        lognorm.fit(samples, floc=0)[0] >= np.log(1.20) and
        lognorm.fit(samples, floc=0)[0] <= np.log(1.30)
    )

    assert (
        lognorm.fit(samples, floc=0)[-1] >= 95 and
        lognorm.fit(samples, floc=0)[-1] <= 105
    )

    assert pdist.distribution().u == u.m**-3
    assert pdist.distribution().m.shape == (1000,)
