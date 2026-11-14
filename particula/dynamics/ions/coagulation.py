"""Prepared collision laws for explicit ions and charged aerosols."""

from dataclasses import dataclass

import numpy as np

from particula.dynamics.ions.condensation import IonCondensationExclusion
from particula.runnable import RunnableABC
from particula.util.constants import ATOMIC_MASS_UNIT


@dataclass(frozen=True)
class AtmosphericCoagulation(RunnableABC):
    """Combine charged aerosol coagulation, ion attachment and recombination.

    Prepared Warp execution only, for unit represented particles. Aerosol pairs
    use ``charged_model``; ion-aerosol pairs use the conductive-air LF2013
    tables at 298.15 K and 101325 Pa, with 150 u positive and 90 u negative
    ions. Opposite free ions recombine; same-sign ions do not cluster. Neutral
    ion-only products are inert until an explicit ``IonNeutralCleanup``.
    Reserved material lanes must contain only ion mass.

    Args:
        positive_species: Positive-ion material index.
        negative_species: Negative-ion material index.
        charged_model: Existing charged Brownian model name.
        omitted: ``reject`` (default), ``lower`` or ``upper`` for repulsive
            attachment rates omitted below the published 1e-15 m^3/s cutoff.
            Bounds are sensitivity assumptions, not extrapolated fit values.
        recombination: Opposite-ion recombination coefficient [m^3/s].

    References:
        López-Yglesias and Flagan (2013), Aerosol Science and Technology,
        47:688–704, https://doi.org/10.1080/02786826.2013.783684.
    """

    positive_species: int
    negative_species: int
    charged_model: str = "hard_sphere"
    omitted: str = "reject"
    recombination: float = 1.7e-12

    def _validate_batch(self, aerosol):
        """Reject unsupported model and physical state before GPU upload."""
        from particula.gpu.kernels._coagulation_config import (
            CoagulationMechanismConfig,
            resolve_coagulation_mechanism_config,
        )

        IonCondensationExclusion(
            self.positive_species, self.negative_species
        )._validate_batch(aerosol)
        # The shared configuration validates the same charged model names.
        resolve_coagulation_mechanism_config(
            CoagulationMechanismConfig(
                mechanisms=("charged_hard_sphere",),
                charged_model=self.charged_model,
            )
        )
        if self.omitted not in ("reject", "lower", "upper"):
            raise ValueError("omitted must be reject, lower or upper")
        if (
            isinstance(self.recombination, bool)
            or not np.isscalar(self.recombination)
            or not np.isfinite(self.recombination)
            or self.recombination < 0
        ):
            raise ValueError("recombination must be finite and nonnegative")
        if np.any(aerosol.environment.temperature != 298.15) or np.any(
            aerosol.environment.pressure != 101325.0
        ):
            raise ValueError("LF2013 requires 298.15 K and 101325 Pa")

    def execute(self, *args, **kwargs):
        """Reject native CPU execution without mutating the aerosol."""
        raise NotImplementedError(
            "AtmosphericCoagulation requires prepared Warp execution"
        )


POSITIVE_ION_MASS = 150 * ATOMIC_MASS_UNIT
NEGATIVE_ION_MASS = 90 * ATOMIC_MASS_UNIT
