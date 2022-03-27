""" testing the coagulation rate
"""
import numpy as np
import pytest
from particula import u
from particula.particle import \
    Particle
from particula.util.coagulation_rate import CoagulationRate
from particula.util.dimensionless_coagulation import full_coag

distribution = Particle(
    cutoff=.9999,
    mode=100e-9,
    nbins=1000,
    nparticles=1e5,
    gsigma=1.25,
).particle_distribution()


fine_distribution = Particle(
    cutoff=.9999,
    mode=100e-9,
    nbins=3000,
    nparticles=1e5,
    gsigma=1.25,
).particle_distribution()


radius = Particle(
    cutoff=.9999,
    mode=100e-9,
    nbins=1000,
    nparticles=1e5,
    gsigma=1.25,
).particle_radius

fine_radius = Particle(
    cutoff=.9999,
    mode=100e-9,
    nbins=3000,
    nparticles=1e5,
    gsigma=1.25,
).particle_radius


kernel = full_coag(
    radius=radius,
    mode=100e-9,
    nbins=1000,
    nparticles=1e5,
    gsigma=1.25,
)

fine_kernel = full_coag(
    radius=fine_radius,
    mode=100e-9,
    nbins=3000,
    nparticles=1e5,
    gsigma=1.25,
)

CoagRate = CoagulationRate(
    distribution=distribution,
    radius=radius,
    kernel=kernel,
)

fine_CoagRate = CoagulationRate(
    distribution=fine_distribution,
    radius=fine_radius,
    kernel=fine_kernel,
)

rads = radius
fine_rads = fine_radius

lnds = distribution
fine_lnds = fine_distribution

kern = kernel
fine_kern = fine_kernel

loss = CoagRate.coag_loss()
fine_loss = fine_CoagRate.coag_loss()

gain = CoagRate.coag_gain()
fine_gain = fine_CoagRate.coag_gain()


def test_kern():
    """ first test the kernel
    """

    assert kern.u == u.m**3/u.s
    assert kern.m.shape == rads.shape + rads.shape
    assert fine_kern.m.shape == fine_rads.shape + fine_rads.shape


def test_loss():
    """ test the loss
    """

    assert loss.u == u.m**-4/u.s
    assert loss.m.shape == rads.shape
    assert loss.m.shape == lnds.shape
    assert fine_loss.m.shape == fine_rads.shape
    assert fine_loss.m.shape == fine_lnds.shape


def test_gain():
    """ test the gain
    """

    assert gain.size == rads.size
    assert fine_gain.size == fine_rads.size
    assert gain.u == u.m**-4/u.s


def test_mass():
    """ test mass conservation
    """

    assert np.trapz((gain - loss)*rads**3, rads).u == u.s**-1
    assert np.trapz((gain - loss)*rads**3, rads) <= 0.2 * u.s**-1
    assert np.trapz(
        (gain - loss).to(u.cm**-4/u.s)*rads**3, rads
    ).m == pytest.approx(0.0)

    assert (
        np.absolute(np.trapz((gain - loss)*rads**3, rads))
        <=
        0.005*np.trapz(gain*rads**3, rads)
    )

    assert (
        np.absolute(np.trapz((gain - loss)*rads**3, rads))
        <=
        0.005*np.trapz(loss*rads**3, rads)
    )


def test_rads():
    """ test radii
    """

    assert rads.u == u.m


def test_res():
    """ testing that mass conservation improves with resolution
    """

    assert (
        np.absolute(np.trapz((gain - loss)*rads**3, rads))
        >=
        np.absolute(np.trapz((fine_gain - fine_loss)*fine_rads**3, fine_rads))
    )
