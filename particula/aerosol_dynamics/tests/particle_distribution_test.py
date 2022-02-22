""" test the particle dist
"""

import numpy as np
from scipy.stats import lognorm
import pytest
from particula.aerosol_dynamics.particle_distribution import \
    ParticleDistribution


def test_rad():
    """ testing the radius
    """

    pdist = ParticleDistribution(
        mode=100,
        nbins=1000,
        nparticles=1e5,
        gsigma=1.25,
    )

    assert 100 >= pdist.rad()[0]
    assert 100 <= pdist.rad()[-1]
    assert 1000 == len(pdist.rad())


def test_lnd():
    """ testing the dist
    """

    pdist = ParticleDistribution(
        mode=100,
        nbins=100,
        nparticles=1e5,
        gsigma=1.25,
    )

    assert 0 <= pdist.lnd()[0]
    assert 0 <= pdist.lnd()[-1]
    assert np.trapz(pdist.lnd(), pdist.rad()) == pytest.approx(.9999, rel=1e-1)
    assert len(pdist.lnd()) == len(pdist.rad())
    # assert lognorm.fit(pdist.lnd()) == 11
    assert lognorm.fit(pdist.lnd())[0] <= 3.75
    assert lognorm.fit(pdist.lnd())[1] <= 1e-5


def test_dist():
    """ test dist properties
    """

    pdist = ParticleDistribution(
        mode=100,
        nbins=1000,
        nparticles=1e5,
        gsigma=1.25,
    )

    dps = pdist.rad()
    wts = pdist.lnd()/np.sum(pdist.lnd())

    samples = np.random.choice(dps, size=len(dps), p=wts)

    assert (
        lognorm.fit(samples, floc=0)[0] >= np.log(1.20) and
        lognorm.fit(samples, floc=0)[0] <= np.log(1.30)
    )

    assert (
        lognorm.fit(samples, floc=0)[-1] >= 95 and
        lognorm.fit(samples, floc=0)[-1] <= 105
    )
