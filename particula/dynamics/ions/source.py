"""Configuration for paired atmospheric ions represented as unit particles."""

from dataclasses import dataclass

import numpy as np

from particula.runnable import RunnableABC


@dataclass(frozen=True)
class IonPairSource(RunnableABC):
    """Inject independent Poisson ion pairs into free resolved slots.

    Prepared Warp execution only. An event injects one particle per polarity,
    with signed elementary charge ±1 and unit represented count. Source rates
    are ion pairs per m^3 per second, not total ions per m^3 per second.
    Material masses are externally supplied; the source ledger records them.

    Args:
        production_rate: Nonnegative ion-pair production rate [m^-3 s^-1].
        positive_species: Positive-ion material lane in ParticleData.
        negative_species: Negative-ion material lane in ParticleData.
        positive_mass: Mass of one positive ion [kg].
        negative_mass: Mass of one negative ion [kg].
    """

    production_rate: float
    positive_species: int
    negative_species: int
    positive_mass: float
    negative_mass: float

    def _validate_batch(self, aerosol):
        """Reject invalid source configuration before allocation or mutation."""
        for name in ("production_rate", "positive_mass", "negative_mass"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not np.isscalar(value)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite scalar")
            if value < 0 or (name != "production_rate" and value == 0):
                raise ValueError(f"{name} has an invalid sign")
        p = aerosol.particles
        if p.distribution_type != "particle_resolved":
            raise ValueError("ion production requires particle_resolved")
        for index in (self.positive_species, self.negative_species):
            if type(index) is not int or not 0 <= index < p.n_species:
                raise ValueError("ion species index is invalid")
        if self.positive_species == self.negative_species:
            raise ValueError("ion polarities require distinct material lanes")
        if np.any((p.concentration != 0) & (p.concentration != 1)):
            raise ValueError("ion production requires unit active counts")

    def execute(self, *args, **kwargs):
        """Reject unsupported CPU execution without changing random streams."""
        raise NotImplementedError(
            "IonPairSource requires prepared Warp execution"
        )
