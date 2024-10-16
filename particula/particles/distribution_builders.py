"""Builds distributions strategies based on the specified representation.

Currently, there are no parameters to set, but this is used for consistency
with other builder patterns in the codebase.
"""

from particula.particles.distribution_strategies import (
    MassBasedMovingBin,
    RadiiBasedMovingBin,
    SpeciatedMassMovingBin,
    ParticleResolvedSpeciatedMass,
)
from particula.abc_builder import BuilderABC


class MassBasedMovingBinBuilder(BuilderABC):
    """Builds a MassBasedMovingBin instance."""

    def __init__(self) -> None:
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> MassBasedMovingBin:
        """Builds a MassBasedMovingBin instance."""
        return MassBasedMovingBin()


class RadiiBasedMovingBinBuilder(BuilderABC):
    """Builds a RadiiBasedMovingBin instance."""

    def __init__(self) -> None:
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> RadiiBasedMovingBin:
        """Builds a RadiiBasedMovingBin instance."""
        return RadiiBasedMovingBin()


class SpeciatedMassMovingBinBuilder(BuilderABC):
    """Builds a SpeciatedMassMovingBin instance."""

    def __init__(self) -> None:
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> SpeciatedMassMovingBin:
        """Builds a SpeciatedMassMovingBin instance."""
        return SpeciatedMassMovingBin()


class ParticleResolvedSpeciatedMassBuilder(BuilderABC):
    """Builds a ParticleResolvedSpeciatedMass instance."""

    def __init__(self) -> None:
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> ParticleResolvedSpeciatedMass:
        return ParticleResolvedSpeciatedMass()
