"""Abstract base class for factories that use builders to create strategies."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

logger = logging.getLogger("particula")

# Define a generic type variable for the strategy type, to get good type hints
BuilderT = TypeVar("BuilderT")
StrategyT = TypeVar("StrategyT")


class StrategyFactoryABC(ABC, Generic[BuilderT, StrategyT]):
    """Abstract base class for strategy factories.

    This class provides a generic interface for creating strategy objects
    using builder objects.

    Methods:
        - get_builders:
            Returns the mapping of strategy types to builder instances.
        - get_strategy:
            Gets the strategy instance for the specified strategy.

    Examples:
        ```py title="Simple Usage"
        my_factory = SomeSpecificFactory()
        strategy_instance = my_factory.get_strategy(
            "example_type", {"param": 123}
        )
        # strategy_instance is now built using "example_type"
        ```

    References:
    - "Factory Method Pattern,"
    [Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)
    """

    @abstractmethod
    def get_builders(self) -> Dict[str, BuilderT]:
        """Retrieve a mapping of strategy types to builder instances.

        Returns:
            dict: A dictionary that maps strategy type names (str) to
            builder instances.

        Examples:
            ```py title="Coagulation Factory Example"
            from particula.coagulation_factory import CoagulationFactory

            builders = CoagulationFactory().get_builders()
            # Example result:
            # {
            #     "brownian": BrownianCoagulationBuilder(),
            #     "charged": ChargedCoagulationBuilder(),
            #     "turbulent_shear": TurbulentShearCoagulationBuilder(),
            #     "turbulent_dns": TurbulentDNSCoagulationBuilder(),
            #     "combine": CombineCoagulationStrategyBuilder(),
            # }
            ```

        References:
            - "Factory Method Pattern,"
            [Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)
        ```
        """

    def get_strategy(
        self, strategy_type: str, parameters: Optional[Dict[str, Any]] = None
    ) -> StrategyT:
        """Create a strategy instance using its corresponding builder.

        Arguments:
            - strategy_type (str): Name of the strategy to build.
            - parameters (Dict[str, Any], optional): Dictionary of parameters
              to configure the chosen builder.

        Returns:
            StrategyT: The built strategy object corresponding to the
                specified type.

        Raises:
            - ValueError: If the `strategy_type` is unknown, or if any required
              parameter is invalid/missing for the chosen builder.

        Examples:
            ```py title="Strategy Creation Example"
            my_factory = SomeStrategyFactory()
            my_strategy = my_factory.get_strategy(
                "desired_strategy", {"param_x": 42}
            )
            # my_strategy is now an instance configured with param_x=42
            ```

        References:
            - "Factory Method Pattern,"
            [Wikipedia](https://en.wikipedia.org/wiki/Factory_method_pattern)
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
