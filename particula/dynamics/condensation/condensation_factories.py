"""Factory for building condensation strategies."""

from typing import Any, Dict

from particula.abc_factory import StrategyFactoryABC
from particula.dynamics.condensation.condensation_builder.condensation_isothermal_builder import (
    CondensationIsothermalBuilder,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationStrategy,
)


class CondensationFactory(
    StrategyFactoryABC[CondensationIsothermalBuilder, CondensationStrategy]
):
    """Factory class for condensation strategies."""

    def get_builders(self) -> Dict[str, Any]:
        return {"isothermal": CondensationIsothermalBuilder()}
