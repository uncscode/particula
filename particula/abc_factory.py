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


class StrategyFactory(ABC, Generic[BuilderT, StrategyT]):
    """
    Abstract base class for strategy factories.
    """

    @abstractmethod
    def get_builders(self) -> Dict[str, BuilderT]:
        """
        Returns the mapping of key names to builders.
        """

    def get_strategy(
        self, strategy_type: str, parameters: Optional[Dict[str, Any]] = None
    ) -> StrategyT:
        """
        Generic factory method to create objects instances.

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
