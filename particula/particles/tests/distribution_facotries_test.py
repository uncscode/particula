"""Testing the distribution factories, for the distribution strategies.

The factories are tested to ensure that the correct distribution strategy is
created based on the factory used.

The Strategy is tested independently.
"""

import pytest
from particula.particles.distribution_factories import (
    DistributionFactory,
)
from particula.particles.distribution_strategies import (
    MassBasedMovingBin,
    RadiiBasedMovingBin,
    SpeciatedMassMovingBin,
    ParticleResolvedSpeciatedMass,
)


def test_mass_based_moving_bin():
    """Test factory function for mass-based moving bin strategy."""
    strategy = DistributionFactory().get_strategy("mass_based_moving_bin")
    assert isinstance(strategy, MassBasedMovingBin)


def test_radii_based_moving_bin():
    """Test factory function for radii-based moving bin strategy."""
    strategy = DistributionFactory().get_strategy("radii_based_moving_bin")
    assert isinstance(strategy, RadiiBasedMovingBin)


def test_speciated_mass_moving_bin():
    """Test factory function for speciated mass moving bin strategy."""
    strategy = DistributionFactory().get_strategy("speciated_mass_moving_bin")
    assert isinstance(strategy, SpeciatedMassMovingBin)


def test_particle_resolved_speciated_mass():
    """Test factory function for particle resolved speciated mass strategy."""
    strategy = DistributionFactory().get_strategy(
        "particle_resolved_speciated_mass"
    )
    assert isinstance(strategy, ParticleResolvedSpeciatedMass)


def test_invalid_strategy():
    """Test factory function for invalid type."""
    with pytest.raises(ValueError) as excinfo:
        DistributionFactory().get_strategy("invalid_type")
    assert "Unknown strategy type: invalid_type" in str(excinfo.value)
