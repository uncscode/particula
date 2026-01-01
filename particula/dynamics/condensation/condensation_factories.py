"""Factory for building condensation strategies."""

from typing import Any, Dict

from particula.abc_factory import StrategyFactoryABC

from .condensation_builder import (
    CondensationIsothermalBuilder,
    CondensationIsothermalStaggeredBuilder,
)
from .condensation_strategies import CondensationStrategy


class CondensationFactory(StrategyFactoryABC[Any, CondensationStrategy]):
    """Factory class for condensation strategies."""

    def get_builders(self) -> Dict[str, Any]:
        """Return the mapping of strategy types to builder instances.

        Returns:
            Dictionary mapping condensation strategy names to builders.
        """
        return {
            "isothermal": CondensationIsothermalBuilder(),
            "isothermal_staggered": CondensationIsothermalStaggeredBuilder(),
        }
