"""Abstract Base Class for Factory classes, that use builders to create
strategy objects.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Generic, TypeVar, Any
import logging

logger = logging.getLogger("particula")

# Define a generic type variable for the strategy type, to get good type hints
BuilderT = TypeVar("BuilderT")
StrategyT = TypeVar("StrategyT")


class StrategyFactoryABC(ABC, Generic[BuilderT, StrategyT]):
    """
    Abstract base class for strategy factories.

    This class provides a generic interface for creating strategy objects
    using builder objects.

    Methods:
    - get_builders : Returns the mapping of strategy types to builder
        instances.
    - get_strategy : Gets the strategy instance for the specified strategy.
        - strategy_type : Type of strategy to use.
        - parameters : Parameters required for the
            builder, dependent on the chosen strategy type.
    """

    @abstractmethod
    def get_builders(self) -> Dict[str, BuilderT]:
        """
        Returns the mapping of key names to builders, and their strategy
        build methods.

        Returns:
            A dictionary mapping strategy types to builder instances.

        Example:
        ``` py title= "Coagulation Factory"
        CoagulationFactory().get_builders()
        # Returns:
            {
                "brownian": BrownianCoagulationBuilder(),
                "charged": ChargedCoagulationBuilder(),
                "turbulent_shear": TurbulentShearCoagulationBuilder(),
                "turbulent_dns": TurbulentDNSCoagulationBuilder(),
                "combine": CombineCoagulationStrategyBuilder(),
            }
        ```
        """

    def get_strategy(
        self, strategy_type: str, parameters: Optional[Dict[str, Any]] = None
    ) -> StrategyT:
        """
        Generic factory method to create objects instances.

        Args:
            - strategy_type : Type of strategy to use.
            - parameters : Parameters required for the
                builder, dependent on the chosen strategy type. Try building
                with a builder first, to see if it is valid.

        Returns:
            An object, built from selected builder with parameters.

        Raises:
            - ValueError : If an unknown key is provided.
            - ValueError : If any required parameter is missing during
                check_keys or pre_build_check, or if trying to set an
                invalid parameter.
        """
        builder_map = self.get_builders()
        builder = builder_map.get(strategy_type.lower())
        if builder is None:
            message = f"Unknown strategy type: {strategy_type}"
            logger.error(message)
            raise ValueError(message)

        if parameters and hasattr(builder, "set_parameters"):
            builder.set_parameters(parameters)  # type: ignore

        return builder.build()  # type: ignore
