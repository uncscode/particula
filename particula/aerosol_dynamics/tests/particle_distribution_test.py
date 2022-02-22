""" test the particle dist
"""

import numpy as np
import pytest
# from particula import u
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
