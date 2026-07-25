"""Define immutable P4 nucleation source-selection metadata.

This module only selects an already-built potential-rate strategy and precursor
species index. It deliberately does not import or provide concrete P2/P3
``particle_source`` finalization, transaction, or state-mutation helpers.
"""

from dataclasses import dataclass

import numpy as np

from particula.dynamics.nucleation.nucleation_strategies import (
    ActivationNucleationStrategy,
    KineticNucleationStrategy,
    NucleationStrategy,
)


@dataclass(frozen=True)
class NucleationSourceConfig:
    """Select a P4 nucleation strategy and its precursor species index.

    This record is metadata only; source-demand finalization and mutation remain
    at concrete P2/P3 boundaries.

    Attributes:
        strategy: Immutable activation or kinetic potential-rate strategy.
        precursor_index: Nonnegative index of the precursor species selected for
            a later concrete source operation.
    """

    strategy: NucleationStrategy
    precursor_index: int

    def __post_init__(self) -> None:
        """Validate the supported strategy types and nonnegative index.

        Raises:
            TypeError: If ``precursor_index`` is not a non-Boolean integer.
            ValueError: If the strategy is unsupported or the index is negative.
        """
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
