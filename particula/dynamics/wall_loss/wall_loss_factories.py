"""Wall loss strategy factory.

Provides a factory for creating wall loss strategies by name using registered
builders.
"""

from typing import Any, Dict, Optional, Union

from particula.abc_factory import StrategyFactoryABC
from particula.dynamics.wall_loss.wall_loss_builders import (
    RectangularWallLossBuilder,
    SphericalWallLossBuilder,
)

BuilderType = Union[SphericalWallLossBuilder, RectangularWallLossBuilder]


class WallLossFactory(StrategyFactoryABC):
    """Factory for wall loss strategies.

    Supported strategy types:
        - "spherical": ``SphericalWallLossStrategy`` for spherical chambers.
        - "rectangular": ``RectangularWallLossStrategy`` for box chambers.
    """

    def get_builders(self) -> Dict[str, BuilderType]:
        """Return available wall loss builders keyed by strategy name."""
        return {
            "spherical": SphericalWallLossBuilder(),
            "rectangular": RectangularWallLossBuilder(),
        }

    def get_strategy(
        self, strategy_type: str, parameters: Optional[Dict[str, Any]] = None
    ):
        """Create a wall loss strategy using its corresponding builder.

        Args:
            strategy_type: Strategy name ("spherical" or "rectangular").
            parameters: Optional parameter mapping to configure the builder.

        Returns:
            Built wall loss strategy instance.

        Raises:
            ValueError: If ``strategy_type`` is unknown.
        """
        builder_map = self.get_builders()
        builder = builder_map.get(strategy_type.lower())
        if builder is None:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        parameter_copy = dict(parameters) if parameters else {}
        distribution_type = parameter_copy.pop("distribution_type", None)

        if parameter_copy:
            builder.set_parameters(parameter_copy)
        if distribution_type is not None:
            builder.set_distribution_type(distribution_type)

        return builder.build()
