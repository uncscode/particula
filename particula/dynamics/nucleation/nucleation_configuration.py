"""Immutable P4 nucleation source-selection configuration."""

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
    """

    strategy: NucleationStrategy
    precursor_index: int

    def __post_init__(self) -> None:
        """Validate the supported strategy types and nonnegative index."""
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
