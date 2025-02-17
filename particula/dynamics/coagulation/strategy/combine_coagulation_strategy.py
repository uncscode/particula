"""
Combines a list of coagulation strategies. The kernel is the sum of the
kernels of the strategies. The distribution type must all be the same.
"""

from typing import List, Union
import logging
import numpy as np
from numpy.typing import NDArray

from particula.particles.representation import ParticleRepresentation
from particula.dynamics.coagulation.strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)

logger = logging.getLogger("particula")


class CombineCoagulationStrategy(CoagulationStrategyABC):
    """
    Combines multiple coagulation strategies into one.

    This class takes a list of coagulation strategies and combines their
    kernels by summing them. All strategies must have the same distribution
    type.

    Arguments:
        - strategies: A list of coagulation strategies to combine.

    Methods:
    - kernel: Calculate the combined coagulation kernel.
    - loss_rate: Calculate the combined coagulation loss rate.
    - gain_rate: Calculate the combined coagulation gain rate.
    - net_rate: Calculate the combined net coagulation rate.
    """

    def __init__(self, strategies: List[CoagulationStrategyABC]):
        if not strategies:
            raise ValueError("At least one strategy must be provided.")

        distribution_type = strategies[0].distribution_type
        for strategy in strategies:
            if strategy.distribution_type != distribution_type:
                raise ValueError("All strategies must have the same distribution type.")

        super().__init__(distribution_type=distribution_type)
        self.strategies = strategies

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        message = (
            "Dimensionless kernel not implemented in combined "
            + "coagulation strategy."
        )
        logger.error(message)
        raise NotImplementedError(message)

    def kernel(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        combined_kernel = 0
        for strategy in self.strategies:
            combined_kernel += strategy.kernel(
                particle=particle, temperature=temperature, pressure=pressure
            )
        return combined_kernel
