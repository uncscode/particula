"""Tests for representation builders."""

import numpy as np

from particula.next.particles.representation_builders import (
    MassParticleRepresentationBuilder,
    RadiusParticleRepresentationBuilder,
    LimitedRadiusParticleBuilder,
)
from particula.next.particles.representation import ParticleRepresentation
from particula.next.particles.distribution_strategies import (
    RadiiBasedMovingBin,
)
from particula.next.particles.surface_strategies import SurfaceStrategyVolume
from particula.next.particles.activity_strategies import IdealActivityMass


def test_mass_particle_representation_builder():
    """Test MassParticleRepresentationBuilder Builds.
    """
    builder = MassParticleRepresentationBuilder()
    builder.set_distribution_strategy(RadiiBasedMovingBin())
    builder.set_activity_strategy(IdealActivityMass())
    builder.set_surface_strategy(SurfaceStrategyVolume())
    builder.set_mass(np.array([1.0, 2.0, 3.0]))
    builder.set_density(np.array([1.0, 2.0, 3.0]))
    builder.set_concentration(np.array([10, 20, 30]))
    builder.set_charge(1.0)
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)


def test_radius_particle_representation_builder():
    """Test RadiusParticleRepresentationBuilder Builds.
    """
    builder = RadiusParticleRepresentationBuilder()
    builder.set_distribution_strategy(RadiiBasedMovingBin())
    builder.set_activity_strategy(IdealActivityMass())
    builder.set_surface_strategy(SurfaceStrategyVolume())
    builder.set_radius(np.array([1.0, 2.0, 3.0]))
    builder.set_density(np.array([1.0, 2.0, 3.0]))
    builder.set_concentration(np.array([10, 20, 30]))
    builder.set_charge(np.array([1.0, 2.0, 3.0]))
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)


def test_limited_radius_particle_builder():
    """Test LimitedRadiusParticleBuilder Builds.
    """
    # default values
    builder = LimitedRadiusParticleBuilder()
    particle_representation_defaults = builder.build()
    assert isinstance(particle_representation_defaults, ParticleRepresentation)

    # set values
    builder = LimitedRadiusParticleBuilder()
    builder.set_mode(np.array([100, 2000]), 'nm')
    builder.set_geometric_standard_deviation(np.array([1.4, 1.5]))
    builder.set_number_concentration(np.array([1e3, 1e3]), 'cm^-3')
    particle_representation = builder.build()
    assert isinstance(particle_representation, ParticleRepresentation)
