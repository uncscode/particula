""" Test the particle distribution class
"""

import numpy as np
import pytest
from scipy.stats import lognorm

from particula import u
from particula.aerosol_dynamics.particle_distribution import \
    ParticleDistribution


def test_rad():
    """ testing the radius properties

        * starting radius is smaller than mean
        * ending radius is larger than mean
        * size of radius is equal to number of bins
    """

    pdist = ParticleDistribution(
        cutoff=.9999,
        mode=100e-9,
        nbins=1000,
        nparticles=1e5,
        gsigma=1.25,
        volume=1e-6,
    )

    pdist_log = ParticleDistribution(
        cutoff=.9999,
        mode=100e-9,
        nbins=1000,
        nparticles=1e5,
        gsigma=1.25,
        volume=1e-6,
        spacing="logspace",
    )

    assert 100 >= pdist.radius()[0].m_as('nm')
    assert 100 <= pdist.radius()[-1].m_as('nm')
    assert 1000 == len(pdist.radius())
    assert 100 >= pdist_log.radius()[0].m_as('nm')
    assert 100 <= pdist_log.radius()[-1].m_as('nm')
    assert 1000 == len(pdist_log.radius())

def test_discretize():
    """ testing the discretization

        * all values of discretization >= 0
        * integral of discretization is equal to 1 (pdf)
        * shape of distribution depends on shape of radius
        * recovering the distribution properties from discretization
    """

    pdist = ParticleDistribution(
        mode=100,
        nbins=100,
        nparticles=1e5,
        gsigma=1.25,
    )

    pdist_log = ParticleDistribution(
        mode=100,
        nbins=100,
        nparticles=1e5,
        gsigma=1.25,
        spacing="logspace",
    )

    assert 0 <= pdist.discretize()[0]
    assert 0 <= pdist.discretize()[-1]
    assert np.trapz(pdist.discretize(), pdist.radius(
    )) == pytest.approx(.9999, rel=1e-1)
    assert len(pdist.discretize().m) == len(pdist.radius().m)
    assert lognorm.fit(pdist.discretize().m)[0] <= 3.75
    assert lognorm.fit(pdist.discretize().m)[1] <= 1e-5

    assert 0 <= pdist_log.discretize()[0]
    assert 0 <= pdist_log.discretize()[-1]
    assert np.trapz(pdist_log.discretize(), pdist_log.radius(
    )) == pytest.approx(.9999, rel=1e-1)
    assert len(pdist_log.discretize().m) == len(pdist_log.radius().m)
    assert lognorm.fit(pdist_log.discretize().m)[0] <= 3.75
    assert lognorm.fit(pdist_log.discretize().m)[1] <= 1e-5


def test_dist():
    """ test distribution properties

        * able to recover the distribution properties from discretization
        * distribution units are okay
        * distribution shape is okay
    """

    pdist = ParticleDistribution(
        mode=100,
        nbins=1000,
        nparticles=1e5,
        gsigma=1.25,
    )

    pdist_log = ParticleDistribution(
        mode=100,
        nbins=1000,
        nparticles=1e5,
        gsigma=1.25,
        spacing="logspace",
    )

    dps = pdist.radius().m
    wts = pdist.discretize()/np.sum(pdist.discretize())

    samples = np.random.choice(dps, size=len(dps), p=wts.m)

    assert (
        lognorm.fit(samples, floc=0)[0] >= np.log(1.20) and
        lognorm.fit(samples, floc=0)[0] <= np.log(1.30)
    )

    assert (
        lognorm.fit(samples, floc=0)[-1] >= 95 and
        lognorm.fit(samples, floc=0)[-1] <= 105
    )

    assert pdist.distribution().u == u.m**-4
    assert pdist.distribution().m.shape == (1000,)

    dps_log = pdist_log.radius().m
    wts_log = pdist_log.discretize()/np.sum(pdist_log.discretize())

    samples_log = np.random.choice(dps_log, size=len(dps_log), p=wts_log.m)

    assert (
        lognorm.fit(samples_log, floc=0)[0] >= np.log(1.20) and
        lognorm.fit(samples_log, floc=0)[0] <= np.log(1.30)
    )

    assert (
        lognorm.fit(samples_log, floc=0)[-1] >= 95 and
        lognorm.fit(samples_log, floc=0)[-1] <= 105
    )

    assert pdist_log.distribution().u == u.m**-4
    assert pdist_log.distribution().m.shape == (1000,)
