"""
CombineCoagulationStrategyBuilder

Creates a combined coagulation strategy from multiple sub-strategies.

This builder follows standard usage:
    builder.set_strategies([...]).build()
"""

from typing import List, Optional
import logging

from particula.abc_builder import BuilderABC
from particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy import (
    CombineCoagulationStrategy,
    CoagulationStrategyABC,
)

logger = logging.getLogger("particula")


class CombineCoagulationStrategyBuilder(BuilderABC):
    """Builder used to create a CombineCoagulationStrategy object.

    Attributes:
        strategies (List[CoagulationStrategyABC]):
            Collection of CoagulationStrategyABC objects to be combined.
    """

    def __init__(self):
        """Initializes the builder with the required parameters."""
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
        """
        self.pre_build_check()
        return CombineCoagulationStrategy(strategies=self.strategies)
