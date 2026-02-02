"""Quick script to demonstrate ParticleDataBuilder usage."""

from __future__ import annotations

import numpy as np

from particula.particles.particle_data_builder import ParticleDataBuilder


def test_particle_data_builder_snippet() -> None:
    """Instantiate the builder and print derived arrays."""

    builder = (
        ParticleDataBuilder()
        .set_masses(np.array([[[1e-18, 2e-18], [1.5e-18, 2.5e-18]]]))
        .set_density(np.array([1500.0, 1800.0]))
        .set_concentration(np.array([1.0, 2.0]))
        .set_charge(np.array([0.0, 0.1]))
        .set_volume(np.array([1.0]))
    )

    particle_data = builder.build()

    print("concentration shape:", particle_data.concentration.shape)
    print("concentration array:", particle_data.concentration)
    print("charge shape:", particle_data.charge.shape)
    print("volume shape:", particle_data.volume.shape)
    print("density shape:", particle_data.density.shape)
