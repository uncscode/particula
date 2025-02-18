"""
CombineCoagulationStrategyBuilder class to create a combined coagulation strategy.
"""

from typing import List

from particula.abc_builder import BuilderABC
from particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy import (
    CombineCoagulationStrategy,
    CoagulationStrategyABC,
)


class CombineCoagulationStrategyBuilder(BuilderABC):
    """
    Builder for creating a CombineCoagulationStrategy instance.
    """

    def __init__(self):
        """
        Initializes the builder with the required parameters.
        """
        required_parameters = ["strategies"]
        super().__init__(required_parameters)
        self.strategies = []

    def set_strategies(self, strategies: List[CoagulationStrategyABC]):
        """
        Set a list of CoagulationStrategyABC-compatible strategies to be combined.

        :param strategies: A list of coagulation strategies to combine.
        """
        self.strategies = strategies
        return self

    def build(self) -> CombineCoagulationStrategy:
        """
        Finalize and return the combined coagulation strategy.

        :return: An instance of CombineCoagulationStrategy.
        """
        self.pre_build_check()
        return CombineCoagulationStrategy(strategies=self.strategies)
