""" Test the particle distribution class
"""

import numpy as np
import pytest
from scipy.stats import lognorm
from particula import u
from particula.particle import Particle

pdist_lin = Particle(
    cutoff=.9999,
    mode=100e-9,
    nbins=10000,
    nparticles=1e5,
    gsigma=1.25,
    volume=1e-6,
    spacing="linspace",
)

pdist_log = Particle(
    cutoff=.9999,
    mode=100e-9,
    nbins=10000,
    nparticles=1e5,
    gsigma=1.25,
    volume=1e-6,
    spacing="logspace",
)


def test_rad():
    """ testing the radius properties

        * starting radius is smaller than mean
        * ending radius is larger than mean
        * size of radius is equal to number of bins
    """

    for pdist in [pdist_lin, pdist_log]:
        assert 100 >= pdist.particle_radius[0].m_as('nm')
        assert 100 <= pdist.particle_radius[-1].m_as('nm')
        assert 10000 == len(pdist.particle_radius)


def test_discretize():
    """ testing the discretization

        * all values of discretization >= 0
        * integral of discretization is equal to 1 (pdf)
        * shape of distribution depends on shape of radius
    """

    for pdist in [pdist_lin, pdist_log]:
        assert 0 <= pdist.pre_discretize()[0]
        assert 0 <= pdist.pre_discretize()[-1]
        assert np.trapz(
            pdist.pre_discretize(), pdist.particle_radius
        ) == pytest.approx(.9999, rel=1e-1)
        assert len(pdist.pre_discretize().m) == len(pdist.particle_radius.m)


def test_dist():
    """ test distribution properties

        * able to recover the distribution properties from discretization
        * distribution units are okay
        * distribution shape is okay
    """

    for pdist in [pdist_lin, pdist_log]:

        dps = pdist.particle_radius.m
        wts = pdist.pre_discretize()/np.sum(pdist.pre_discretize())
        samples = np.random.choice(dps, size=len(dps), p=wts.m)

        assert (
            lognorm.fit(samples, floc=0)[0] >= np.log(1.20) and
            lognorm.fit(samples, floc=0)[0] <= np.log(1.30)
        )

        assert (
            lognorm.fit(samples, floc=0)[-1] >= 90e-9 and
            lognorm.fit(samples, floc=0)[-1] <= 110e-9
        )

        assert pdist.particle_distribution().u == u.m**-4
        assert pdist.particle_distribution().m.shape == (10000,)
