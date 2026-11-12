"""Define immutable public construction nucleation source-selection metadata.

This module only selects an already-built potential-rate strategy and precursor
species index, with an explicit full-gas map for native execution. It does not
import or provide concrete demand planning/source transaction
``particle_source`` finalization, transaction, or state-mutation helpers.
"""

from dataclasses import dataclass

import numpy as np

from particula.dynamics.nucleation.nucleation_strategies import (
    ActivationNucleationStrategy,
    KineticNucleationStrategy,
    NucleationStrategy,
)
from particula.dynamics.nucleation.species_mapping import NucleationSpeciesMap


@dataclass(frozen=True)
class NucleationSourceConfig:
    """Select a public construction nucleation strategy and its precursor
        species index.

    This record is metadata only; source-demand finalization and mutation remain
    at concrete demand planning/source transaction boundaries.

    Attributes:
        strategy: Immutable activation or kinetic potential-rate strategy.
        precursor_index: Nonnegative index of the precursor species selected for
            a later concrete source operation.
        species_map: Explicit native gas order and injection mapping; None is
            retained only for transitional legacy callers.
    """

    strategy: NucleationStrategy
    precursor_index: int
    species_map: NucleationSpeciesMap | None = None

    def __post_init__(self) -> None:
        """Validate the supported strategy types and nonnegative index.

        Raises:
            TypeError: If ``precursor_index`` is not a non-Boolean integer.
            ValueError: If the strategy is unsupported or the index is negative.
        """
        if self.species_map is not None and not isinstance(
            self.species_map, NucleationSpeciesMap
        ):
            raise TypeError("species_map must be NucleationSpeciesMap")
        if not isinstance(
            self.strategy,
            (ActivationNucleationStrategy, KineticNucleationStrategy),
        ):
            raise ValueError(
                "strategy must be ActivationNucleationStrategy or "
                "KineticNucleationStrategy"
            )
        if not isinstance(
            self.precursor_index, (int, np.integer)
        ) or isinstance(self.precursor_index, (bool, np.bool_)):
            raise TypeError("precursor_index must be an integer")
        if self.precursor_index < 0:
            raise ValueError("precursor_index must be nonnegative")
