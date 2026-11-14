"""Opt-in original-seed lineage observations for prepared resolved studies."""

from dataclasses import dataclass

import numpy as np

from particula.runnable import RunnableABC


@dataclass(frozen=True)
class SeedCohort(RunnableABC):
    """Track original seed survival to a diameter before background capture.

    Initially, each unit-count particle containing seed material contributes
    one lineage. Particles containing background material carry a persistent
    background label. Seed-seed mergers add unresolved lineages; any background
    merger captures unresolved lineages before evaluating the resulting size.
    Successful lineages are counted once and remain successful thereafter.

    This prepared-only observer runs after each physical process, independently
    of its position in the composition. Events follow the configured process
    order; substep crossing times are not interpolated. Unresolved lineages are
    right-censored when the user ends the observation.

    Args:
        seed_species: Initial seed-marker material index.
        background_species: Initial background-marker material index.
        target_diameter: First-passage diameter [m].
    """

    seed_species: int
    background_species: int
    target_diameter: float = 50e-9

    def _validate_batch(self, aerosol):
        """Require an explicit material map and resolved unit counts."""
        p = aerosol.particles
        if p.distribution_type != "particle_resolved":
            raise ValueError("seed cohorts require particle_resolved")
        for index in (self.seed_species, self.background_species):
            if type(index) is not int or not 0 <= index < p.n_species:
                raise ValueError("cohort species index is invalid")
        if self.seed_species == self.background_species:
            raise ValueError("seed and background materials must be distinct")
        if np.any((p.concentration != 0) & (p.concentration != 1)):
            raise ValueError("seed cohorts require unit active counts")
        if (
            isinstance(self.target_diameter, bool)
            or not np.isscalar(self.target_diameter)
            or not np.isfinite(self.target_diameter)
            or self.target_diameter <= 0
        ):
            raise ValueError("target diameter must be finite and positive")
        volume = np.pi / 6 * float(self.target_diameter) ** 3
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("target volume must be representable")

    def execute(self, *args, **kwargs):
        """Reject unsupported native CPU execution explicitly."""
        raise NotImplementedError("SeedCohort requires prepared Warp execution")
