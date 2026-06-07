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
    _normalize_resolved_charge_input,
    _normalize_resolved_density_input,
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


def test_resolved_mass_builder_flattens_singleton_density_and_charge() -> None:
    """Singleton 2D density/charge inputs should remain build-compatible."""
    builder = ResolvedParticleMassRepresentationBuilder()
    builder.set_distribution_strategy(ParticleResolvedSpeciatedMass())
    builder.set_activity_strategy(ActivityIdealMass())
    builder.set_surface_strategy(SurfaceStrategyVolume())
    builder.set_mass(np.array([[1.0], [2.0], [3.0]]), "kg")
    builder.set_density(np.full((3, 1), 1000.0), "kg/m^3")
    builder.set_charge(np.zeros((3, 1)))
    builder.set_volume(1, "m^3")

    particle_representation = builder.build()

    assert isinstance(particle_representation, ParticleRepresentation)
    np.testing.assert_array_equal(
        particle_representation.get_density(), np.array([1000.0])
    )
    np.testing.assert_array_equal(
        particle_representation.get_charge(), np.zeros(3)
    )


def test_resolved_mass_builder_flattens_singleton_row_charge() -> None:
    """Singleton-row charge arrays should flatten across all particles."""
    builder = ResolvedParticleMassRepresentationBuilder()
    builder.set_distribution_strategy(ParticleResolvedSpeciatedMass())
    builder.set_activity_strategy(ActivityIdealMass())
    builder.set_surface_strategy(SurfaceStrategyVolume())
    builder.set_mass(np.array([[1.0], [2.0], [3.0]]), "kg")
    builder.set_density(np.array([[1000.0], [1000.0], [1000.0]]), "kg/m^3")
    builder.set_charge(np.array([[1.0, 2.0, 3.0]]))
    builder.set_volume(1, "m^3")

    particle_representation = builder.build()

    assert isinstance(particle_representation, ParticleRepresentation)
    np.testing.assert_array_equal(
        particle_representation.get_charge(),
        np.array([1.0, 2.0, 3.0]),
    )


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


def test_normalize_resolved_density_input_handles_matching_and_empty_shapes(
) -> None:
    """Resolved density normalization should reduce to the species axis."""
    mass = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    np.testing.assert_array_equal(
        _normalize_resolved_density_input(
            np.array([[1000.0, 900.0], [1000.0, 900.0]]),
            mass,
        ),
        np.array([1000.0, 900.0]),
    )
    np.testing.assert_array_equal(
        _normalize_resolved_density_input(
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
        ),
        np.zeros(2, dtype=np.float64),
    )


def test_normalize_resolved_charge_input_handles_none_matching_and_singleton(
) -> None:
    """Resolved charge normalization should reduce to the particle axis."""
    mass = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    assert _normalize_resolved_charge_input(None, mass) is None
    np.testing.assert_array_equal(
        _normalize_resolved_charge_input(
            np.array([[1.0, 9.0], [2.0, 8.0]]),
            mass,
        ),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_array_equal(
        _normalize_resolved_charge_input(
            np.array([[1.0], [2.0]]),
            mass,
        ),
        np.array([1.0, 2.0]),
    )
