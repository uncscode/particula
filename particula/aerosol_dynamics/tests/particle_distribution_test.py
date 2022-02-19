"""Test suites for particle class.
"""

import numpy as np
import pytest
from particula.aerosol_dynamics import particle_distribution

SIZE = 10000
radii_sample = np.random.poisson(100, size=SIZE)*10**-9  # m

particle_dist = particle_distribution.ParticleDistribution(
        radii=radii_sample,
        density=np.ones(len(radii_sample))*1000,
        charge=np.zeros(len(radii_sample)),
        number=np.ones(len(radii_sample)),
    )


def test_getters():
    """
    Tests the getters for the Environment class.
    """
    assert particle_dist.name() == 'Distribution'
    assert len(particle_dist.radii()) == SIZE
    assert len(particle_dist.densities()) == SIZE
    assert len(particle_dist.charges()) == SIZE
    assert len(particle_dist.number()) == SIZE


def distribution_properties():
    """
    Test the calculation of total concentration properties
    """
    assert len(particle_dist.masses()) == SIZE
    assert particle_dist.number_concentration() == SIZE
    assert particle_dist.mass_concentration() > 0


def rasterization_process():
    """
    Test the rasterization process
    """
    old_number = particle_dist.number_concentration()
    old_mass = particle_dist.mass_concentration()

    particle_dist.rasterization_and_update('auto')

    assert particle_dist.number_concentration() <= old_number
    assert (
        particle_dist.mass_concentration() ==
        pytest.approx(old_mass, rel=1e-1)
    )
