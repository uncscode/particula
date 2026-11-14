"""Export neutral products made only from reserved ion materials."""

from dataclasses import dataclass

import numpy as np

from particula.runnable import RunnableABC


@dataclass(frozen=True)
class IonNeutralCleanup(RunnableABC):
    """Export neutral ion-only products and reclaim their resolved slots.

    The two material lanes must be reserved for ions. A neutral particle with
    any mass in another lane is an aerosol and is never removed. Both reserved
    lanes must contain positive mass. This operation is a modeled export of
    recombination products, not an additional recombination reaction.

    Prepared execution only; place this process at the desired refresh point
    in the ordered runnable composition. It runs once per prepared step.

    Args:
        positive_species: Reserved positive-ion material index.
        negative_species: Reserved negative-ion material index.
    """

    positive_species: int
    negative_species: int

    def _validate_batch(self, aerosol):
        """Reject incompatible material maps and represented populations."""
        particles = aerosol.particles
        if particles.distribution_type != "particle_resolved":
            raise ValueError("ion cleanup requires particle_resolved")
        for index in (self.positive_species, self.negative_species):
            if type(index) is not int or not 0 <= index < particles.n_species:
                raise ValueError("ion species index is invalid")
        if self.positive_species == self.negative_species:
            raise ValueError("ion polarities require distinct material lanes")
        counts = particles.concentration
        if np.any((counts != 0) & (counts != 1)):
            raise ValueError("ion cleanup requires unit active counts")

    def execute(self, *args, **kwargs):
        """Reject unsupported native CPU execution explicitly."""
        raise NotImplementedError(
            "IonNeutralCleanup requires prepared Warp execution"
        )
