"""Combine Coagulation Strategy Builder Module.

Provides a builder for creating `CombineCoagulationStrategy` objects,
which can merge multiple sub-strategies (e.g., Brownian, Turbulent)
into a single, combined coagulation approach. This allows flexible,
modular composition of different coagulation mechanisms.

Creates a combined coagulation strategy from multiple sub-strategies.

This builder follows standard usage:
    builder.set_strategies([...]).build()
"""

import logging
from typing import List, Optional

from particula.abc_builder import BuilderABC

from ..coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from ..coagulation_strategy.combine_coagulation_strategy import (
    CombineCoagulationStrategy,
)

logger = logging.getLogger("particula")


class CombineCoagulationStrategyBuilder(BuilderABC):
    """Builder for a combined coagulation strategy.

    This class constructs a `CombineCoagulationStrategy` from multiple
    sub-strategies (instances of `CoagulationStrategyABC`), enabling
    advanced modeling scenarios where different coagulation mechanisms
    act concurrently. Each sub-strategy's rate calculations are effectively
    merged to act on the same particle population.

    Attributes:
        - strategies : List of `CoagulationStrategyABC` objects to combine.

    Methods:
    - set_strategies : Set the list of coagulation strategies to combine.
    - build : Create and return the combined coagulation strategy.

    Examples:
        ```py title="Combine Coagulation Strategy Example"
        import particula as par
        builder = par.dynamics.CombineCoagulationStrategyBuilder()
        builder.set_strategies([brownian_strategy, turbulent_strategy])
        combined_strategy = builder.build()
        ```
    """

    def __init__(self):
        """Initialize the CombineCoagulationStrategyBuilder.

        Returns:
            - None

        Note:
            The only required parameter is 'strategies'. Attempting to
            build without setting it triggers an error. Use `set_strategies`
            before calling `build`.
        """
        required_parameters = ["strategies"]
        super().__init__(required_parameters)
        self.strategies = []

    def set_strategies(
        self,
        strategies: List[CoagulationStrategyABC],
        strategies_units: Optional[str] = None,
    ):
        """Sets a list of CoagulationStrategyABC objects to be combined.

        Args:
            strategies : A list of coagulation strategies to be combined.
            strategies_units : For interface consistency, not used.

        Examples:
            ```py title="Set Strategies Example"
            builder = CombineCoagulationStrategyBuilder()
            builder.set_strategies([brownian_strategy, turbulent_strategy])
            ```

        Returns:
            CombineCoagulationStrategyBuilder:
                The builder instance, for fluent chaining.
        """
        if strategies_units is not None:
            logger.warning(
                "The units of the strategies are not used in the "
                "CombineCoagulationStrategyBuilder."
            )
        self.strategies = strategies
        return self

    def build(self) -> CombineCoagulationStrategy:
        """Builds and returns the combined coagulation strategy.

        Returns:
            CombineCoagulationStrategy :
                A strategy that combines all the previously added
                sub-strategies.

        Examples:
            ```py title="Build Example with CombineCoagulationStrategy"
            combined_strategy = builder.build()
            # Now you can use `combined_strategy.kernel(...)` to calculate
            # combined coagulation effects from each sub-strategy.
            ```
        """
        self.pre_build_check()
        return CombineCoagulationStrategy(strategies=self.strategies)
