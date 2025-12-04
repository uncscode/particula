"""Combine multiple coagulation strategies into a single, aggregated approach.

The kernel from each strategy is summed to form a unified coagulation
kernel. All strategies must share the same distribution type.

Classes:
    - CombineCoagulationStrategy : Inherits from CoagulationStrategyABC
      and aggregates multiple strategies.

"""

import logging
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from particula.particles.representation import ParticleRepresentation

from .coagulation_strategy_abc import CoagulationStrategyABC

logger = logging.getLogger("particula")


class CombineCoagulationStrategy(CoagulationStrategyABC):
    """Combine multiple coagulation strategies into one.

    This class takes a list of coagulation strategies and merges their
    kernels by summing them. Each included strategy must share the same
    distribution type.

    Attributes:
        distribution_type : Matches the distribution_type of the first
        strategy.
        strategies : A list of individual CoagulationStrategyABC
        instances.

    Methods:
    - dimensionless_kernel : Raises NotImplementedError, as not supported here.
    - kernel : Compute the sum of all strategy kernels.
    - loss_rate : Calculate the coagulation loss rate.
    - gain_rate : Calculate the coagulation gain rate.
    - net_rate : Get the net coagulation rate (gain - loss).
    - step : Perform a single step of coagulation.
    - diffusive_knudsen : Calculate the diffusive Knudsen number.
    - coulomb_potential_ratio : Compute Coulomb potential ratio.
    - friction_factor : Compute the effective friction factor.


    Examples:
        ```py title="Example Usage of CombineCoagulationStrategy"
        import particula as par
        combined_strategy = par.dynamics.CombineCoagulationStrategy(
            [strategy1, strategy2]
        )
        k_value = combined_strategy.kernel(
            particle=aerosol, temperature=300, pressure=101325
        ) # combined kernel value
        ```

    References:
        - No specific references. Summation approach is straightforward.
    """

    def __init__(self, strategies: List[CoagulationStrategyABC]):
        """Initialize the combined coagulation strategy.

        Arguments:
            - strategies : A list of CoagulationStrategyABC instances to
              combine.

        Returns:
            - None

        Raises:
            - ValueError : If no strategies are provided or if they have
              mismatched distribution types.
        """
        if not strategies:
            raise ValueError("At least one strategy must be provided.")

        distribution_type = strategies[0].distribution_type
        for strategy in strategies:
            if strategy.distribution_type != distribution_type:
                raise ValueError(
                    "All strategies must have the same distribution type."
                )

        super().__init__(distribution_type=distribution_type)
        self.strategies = strategies

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Raise NotImplementedError for dimensionless kernel in combined
        strategy.

        Dimensionless kernels must be handled individually by the underlying
        strategies. This method logs an error and raises NotImplementedError.

        Arguments:
            - diffusive_knudsen : The diffusive Knudsen number(s)
              [dimensionless].
            - coulomb_potential_ratio : The Coulomb potential ratio(s)
              [dimensionless].

        Raises:
            - NotImplementedError : This method is not supported in the
              combined strategy.
        """
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
        """Compute the total coagulation kernel by summing the kernels from all
        underlying strategies.

        Arguments:
            - particle : The ParticleRepresentation instance containing
              particle data (radii, distribution, etc.).
            - temperature : The temperature in Kelvin [K].
            - pressure : The pressure in Pascals [Pa].

        Returns:
            - float or NDArray[np.float64] : The combined coagulation kernel,
              equal to the sum of each strategy's kernel.

        Examples:
            ```py
            k_combined = combined_strategy.kernel(
                particle=my_particle,
                temperature=300.0,
                pressure=101325
            )
            # k_combined is the sum of kernels from each strategy in
            # combined_strategy
            ```
        """
        # Type narrowing: initialize as Union type to match return signature
        combined_kernel: Union[float, NDArray[np.float64]] = 0.0
        for strategy in self.strategies:
            combined_kernel += strategy.kernel(
                particle=particle, temperature=temperature, pressure=pressure
            )
        return combined_kernel
