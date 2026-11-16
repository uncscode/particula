"""Independent additive-volume mixture reference for Stage 1."""

import numpy as np
from numpy.testing import assert_allclose
from particula.particles.particle_data import ParticleData


def test_equal_material_volumes_have_volume_weighted_density():
    """Equal volumes give 1500 kg/m3, including unequal material masses."""
    # Each material occupies 1e-21 m3; the total mass is 3e-18 kg.
    data = ParticleData(
        masses=np.array([[[1e-18, 2e-18], [0.0, 0.0]]]),
        concentration=np.array([[1.0, 0.0]]),
        charge=np.zeros((1, 2)),
        density=np.array([1000.0, 2000.0]),
        volume=np.array([1e-6]),
        distribution_type="particle_resolved",
    )
    assert_allclose(data.effective_density, [[1500.0, 0.0]], rtol=1e-14)
    assert_allclose(
        data.effective_density * (4 * np.pi / 3) * data.radii**3,
        data.total_mass,
        rtol=1e-14,
        atol=0,
    )
