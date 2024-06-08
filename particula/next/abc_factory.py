"""Abstract Base Class for Factory classes, that use builders to create
strategy objects.

Note: Not sure on this approach, we'll see how it grows and if it is useful.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Generic, TypeVar, Any
import logging

logger = logging.getLogger("particula")

# Define a generic type variable for the strategy type, to get good type hints
BuilderT = TypeVar('BuilderT')
StrategyT = TypeVar('StrategyT')


class StrategyFactory(ABC, Generic[BuilderT, StrategyT]):
    """
    Abstract base class for strategy factories.
    """

    @abstractmethod
    def get_builders(self) -> Dict[str, BuilderT]:
        """
        Returns the mapping of strategy types to builder instances.
        """

    def get_strategy(
        self, strategy_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> StrategyT:
        """
        Generic factory method to create strategies.
        """
        builder_map = self.get_builders()
        builder = builder_map.get(strategy_type.lower())
        if builder is None:
            message = f"Unknown strategy type: {strategy_type}"
            logger.error(message)
            raise ValueError(message)

        if parameters and hasattr(builder, 'set_parameters'):
            builder.set_parameters(parameters)  # type: ignore

        return builder.build()  # type: ignore
