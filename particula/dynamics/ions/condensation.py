"""Explicit prepared condensation exclusion for reserved ion-only particles."""

from dataclasses import dataclass

import numpy as np

from particula.runnable import RunnableABC


@dataclass(frozen=True)
class IonCondensationExclusion(RunnableABC):
    """Prevent vapor transfer to or from particles made only of ion materials.

    Pair this configuration with prepared condensation. The two material lanes
    must be reserved for nonvolatile ion mass. Aerosols containing any other
    material remain eligible, including particles carrying attached ion mass.
    Every condensation substep gates proposals before gas inventory limiting.
    Native masses, represented counts and charges are never masked or replaced.

    Args:
        positive_species: Reserved positive-ion material index.
        negative_species: Reserved negative-ion material index.
    """

    positive_species: int
    negative_species: int

    def _validate_batch(self, aerosol):
        """Require resolved unit counts and distinct reserved ion lanes."""
        particles = aerosol.particles
        if particles.distribution_type != "particle_resolved":
            raise ValueError("ion exclusion requires particle_resolved")
        indices = (self.positive_species, self.negative_species)
        if any(
            type(i) is not int or not 0 <= i < particles.n_species
            for i in indices
        ):
            raise ValueError("ion species index is invalid")
        if indices[0] == indices[1]:
            raise ValueError("ion polarities require distinct material lanes")
        counts = particles.concentration
        if np.any((counts != 0) & (counts != 1)):
            raise ValueError("ion exclusion requires unit active counts")

    def execute(self, *args, **kwargs):
        """Reject unsupported native CPU execution explicitly."""
        raise NotImplementedError(
            "IonCondensationExclusion requires prepared Warp execution"
        )
