"""Tests for the distribution builders module.

The builders are tested to ensure that the correct distribution strategy is
created based on the builder used.

The Strategy is tested independently.
"""

from particula.particles.distribution_builders import (
    MassBasedMovingBinBuilder,
    RadiiBasedMovingBinBuilder,
    SpeciatedMassMovingBinBuilder,
    ParticleResolvedSpeciatedMassBuilder,
)
from particula.particles.distribution_strategies import (
    MassBasedMovingBin,
    RadiiBasedMovingBin,
    SpeciatedMassMovingBin,
    ParticleResolvedSpeciatedMass,
)


def test_mass_based_moving_bin_builder():
    """Test returning the MassBasedMovingBin."""
    builder = MassBasedMovingBinBuilder()
    distribution = builder.build()
    assert isinstance(distribution, MassBasedMovingBin)


def test_radii_based_moving_bin_builder():
    """Test returning the RadiiBasedMovingBin."""
    builder = RadiiBasedMovingBinBuilder()
    distribution = builder.build()
    assert isinstance(distribution, RadiiBasedMovingBin)


def test_speciated_mass_moving_bin_builder():
    """Test returning the SpeciatedMassMovingBin."""
    builder = SpeciatedMassMovingBinBuilder()
    distribution = builder.build()
    assert isinstance(distribution, SpeciatedMassMovingBin)


def test_particle_resolved_speciated_mass_builder():
    """Test returning the ParticleResolvedSpeciatedMass."""
    builder = ParticleResolvedSpeciatedMassBuilder()
    distribution = builder.build()
    assert isinstance(distribution, ParticleResolvedSpeciatedMass)
