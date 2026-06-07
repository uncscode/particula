"""Regression tests for zero-particle representations."""

import numpy as np

from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.distribution_strategies import ParticleResolvedSpeciatedMass
from particula.particles.representation_builders import (
    ResolvedParticleMassRepresentationBuilder,
)
from particula.particles.surface_strategies import SurfaceStrategyVolume


def test_resolved_mass_builder_handles_zero_particles_with_array_charge():
    """Zero-particle resolved builds should not warn or fail on charge shape."""
    masses = np.zeros((0, 1), dtype=np.float64)

    particle = (
        ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(ParticleResolvedSpeciatedMass())
        .set_activity_strategy(ActivityIdealMass())
        .set_surface_strategy(SurfaceStrategyVolume())
        .set_mass(masses, "kg")
        .set_density(np.full_like(masses, 1000.0), "kg/m^3")
        .set_charge(np.zeros_like(masses))
        .set_volume(1.0, "m^3")
        .build()
    )

    assert particle.get_mass().size == 0
    assert particle.get_charge().size == 0
