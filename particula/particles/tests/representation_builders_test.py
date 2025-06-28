"""Tests for representation builders."""

import numpy as np

from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.distribution_strategies import (
    ParticleResolvedSpeciatedMass,
    RadiiBasedMovingBin,
)
from particula.particles.representation import ParticleRepresentation
from particula.particles.representation_builders import (
    ParticleMassRepresentationBuilder,
    ParticleRadiusRepresentationBuilder,
    PresetParticleRadiusBuilder,
    PresetResolvedParticleMassBuilder,
    ResolvedParticleMassRepresentationBuilder,
)
from particula.particles.surface_strategies import SurfaceStrategyVolume


def test_mass_particle_representation_builder():
    """Test MassParticleRepresentationBuilder Builds."""
    builder = ParticleMassRepresentationBuilder()
    builder.set_distribution_strategy(RadiiBasedMovingBin())
    builder.set_activity_strategy(ActivityIdealMass())
    builder.set_surface_strategy(SurfaceStrategyVolume())
    builder.set_mass(np.array([1.0, 2.0, 3.0]), "kg")
    builder.set_density(np.array([1.0, 2.0, 3.0]), "kg/m^3")
    builder.set_concentration(np.array([10, 20, 30]), "1/m^3")
    builder.set_charge(1.0)
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)


def test_particle_radius_representation_builder():
    """Test RadiusParticleRepresentationBuilder Builds."""
    builder = ParticleRadiusRepresentationBuilder()
    builder.set_distribution_strategy(RadiiBasedMovingBin())
    builder.set_activity_strategy(ActivityIdealMass())
    builder.set_surface_strategy(SurfaceStrategyVolume())
    builder.set_radius(np.array([1.0, 2.0, 3.0]), "m")
    builder.set_density(np.array([1.0, 2.0, 3.0]), "kg/m^3")
    builder.set_concentration(np.array([10, 20, 30]), "1/m^3")
    builder.set_charge(np.array([1.0, 2.0, 3.0]))
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)


def test_limited_particle_radius_builder():
    """Test LimitedRadiusParticleBuilder Builds."""
    # default values
    builder = PresetParticleRadiusBuilder()
    particle_representation_defaults = builder.build()
    assert isinstance(particle_representation_defaults, ParticleRepresentation)

    # set values
    builder = PresetParticleRadiusBuilder()
    builder.set_mode(np.array([100, 2000]) * 1e-9, "m")
    builder.set_geometric_standard_deviation(np.array([1.4, 1.5]))
    builder.set_number_concentration(np.array([1e3, 1e3]) * 1e6, "1/m^3")
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)


def test_resolved_mass_particle_representation_builder():
    """Test ResolvedMassParticleRepresentationBuilder Builds."""
    builder = ResolvedParticleMassRepresentationBuilder()
    builder.set_distribution_strategy(ParticleResolvedSpeciatedMass())
    builder.set_activity_strategy(ActivityIdealMass())
    builder.set_surface_strategy(SurfaceStrategyVolume())
    builder.set_mass(np.array([1.0, 2.0, 3.0]), "kg")
    builder.set_density(np.array([1.0, 2.0, 3.0]), "kg/m^3")
    builder.set_charge(1.0)
    builder.set_volume(1, "m^3")
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)


def test_preset_resolved_mass_particle_builder():
    """Test PresetResolvedMassParticleBuilder Builds."""
    # default values
    builder = PresetResolvedParticleMassBuilder()
    particle_representation_defaults = builder.build()
    assert isinstance(particle_representation_defaults, ParticleRepresentation)

    # set values
    builder = PresetResolvedParticleMassBuilder()
    builder.set_mode(np.array([100, 2000]) * 1e-9, "m")
    builder.set_geometric_standard_deviation(np.array([1.4, 1.5]))
    builder.set_particle_resolved_count(1000)
    builder.set_volume(1.0, "m^3")
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)
