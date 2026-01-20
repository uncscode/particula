"""Factory for creating equilibria strategies by name.

Provides a registry-based factory that returns equilibria strategies using
co-located builders. Keys are case-insensitive and follow the patterns used
across particula factories.

Examples:
    >>> factory = EquilibriaFactory()
    >>> strategy = factory.get_strategy("liquid_vapor")
    >>> strategy.water_activity
    0.5

    >>> strategy = factory.get_strategy(
    ...     "liquid_vapor", parameters={"water_activity": 0.8}
    ... )
    >>> strategy.water_activity
    0.8
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from particula.abc_factory import StrategyFactoryABC
from particula.equilibria.equilibria_builders import (
    LiquidVaporPartitioningBuilder,
)
from particula.equilibria.equilibria_strategies import EquilibriaStrategy

BuilderType = LiquidVaporPartitioningBuilder


class EquilibriaFactory(StrategyFactoryABC[BuilderType, EquilibriaStrategy]):
    """Factory for equilibria strategies.

    Builds equilibria strategies using registered builders and the
    :class:`StrategyFactoryABC` workflow.

    Supported strategy types:
        - "liquid_vapor": ``LiquidVaporPartitioningStrategy`` for organic
          aerosol liquid-vapor partitioning equilibrium.

    Examples:
        >>> factory = EquilibriaFactory()
        >>> strategy = factory.get_strategy("liquid_vapor")
        >>> strategy.water_activity
        0.5

        >>> strategy = factory.get_strategy(
        ...     "liquid_vapor", parameters={"water_activity": 0.75}
        ... )
        >>> strategy.water_activity
        0.75
    """

    def get_builders(self) -> Dict[str, BuilderType]:
        """Return available equilibria builders keyed by strategy name.

        Returns:
            Dict[str, BuilderType]: Mapping of strategy type to builder.

        Examples:
            >>> factory = EquilibriaFactory()
            >>> builders = factory.get_builders()
            >>> "liquid_vapor" in builders
            True
        """
        return {
            "liquid_vapor": LiquidVaporPartitioningBuilder(),
        }

    def get_strategy(
        self,
        strategy_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> EquilibriaStrategy:
        """Create an equilibria strategy using its corresponding builder.

        Args:
            strategy_type: Name of the strategy to build. Case-insensitive.
                Supported: "liquid_vapor".
            parameters: Optional parameter mapping for the builder.
                For "liquid_vapor": {"water_activity": float}.

        Returns:
            Built equilibria strategy instance.

        Raises:
            ValueError: If ``strategy_type`` is unknown or parameters are
                invalid.
        """
        builder_map = self.get_builders()
        builder = builder_map.get(strategy_type.lower())

        if builder is None:
            valid_types = sorted(builder_map.keys())
            raise ValueError(
                f"Unknown strategy type: '{strategy_type}'. "
                f"Valid types: {valid_types}"
            )

        parameter_copy = dict(parameters) if parameters else {}
        if parameter_copy:
            builder.set_parameters(parameter_copy)

        return builder.build()
