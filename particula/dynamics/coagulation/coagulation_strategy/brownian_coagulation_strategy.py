"""Brownian coagulation strategy class."""

import logging
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state,
)
from particula.particles.representation import ParticleRepresentation

from .coagulation_strategy_abc import CoagulationStrategyABC

logger = logging.getLogger("particula")


class BrownianCoagulationStrategy(CoagulationStrategyABC):
    """Discrete Brownian coagulation strategy class for aerosol simulations.

    This class implements methods defined in CoagulationStrategyABC
    to simulate Brownian coagulation in particle populations. It calculates
    coagulation rates via a Brownian kernel that depends on properties such
    as temperature, pressure, and particle radius.

    Attributes:
        - distribution_type : Defines how particles are represented
          (e.g., "discrete", "continuous_pdf", or "particle_resolved").

    Methods:
    - kernel : Calculate the Brownian coagulation kernel (dimensioned).
    - loss_rate : Calculate the coagulation loss rate (not shown here).
    - gain_rate : Calculate the coagulation gain rate (not shown here).
    - net_rate : Calculate the net coagulation rate (not shown here).
    - dimensionless_kernel : Not implemented, raises NotImplementedError.
    - step : Perform a single step of coagulation.
    - diffusive_knudsen : Calculate the diffusive Knudsen number.
    - coulomb_potential_ratio : Compute Coulomb potential ratio.
    - friction_factor : Compute the effective friction factor.

    Examples:
        ```py title="Example Usage of BrownianCoagulationStrategy"
        import particula as par
        brownian_strat = par.dynamics.BrownianCoagulationStrategy(
            distribution_type="discrete"
        )
        # Suppose we have a ParticleRepresentation object called 'particle_rep'
        # kernel_values = brownian_strat.kernel(
        #   particle_rep, temperature=298, pressure=101325
        # )
        # ...
        ```

    References:
        - `get_brownian_kernel_via_system_state`
        - Seinfeld, J. H., & Pandis, S. N. (2016). "Atmospheric chemistry
          and physics," Section 13, Table 13.1.
    """

    def __init__(self, distribution_type: str):
        """Initialize the BrownianCoagulationStrategy.

        Arguments:
            - distribution_type : String specifying the distribution type
              (e.g., "discrete", "continuous_pdf", "particle_resolved").
        """
        super().__init__(distribution_type=distribution_type)

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Not implemented for BrownianCoagulationStrategy.

        This method raises NotImplementedError since dimensionless
        Brownian kernels are not defined here.

        Arguments:
            - diffusive_knudsen : Knudsen number array (unused).
            - coulomb_potential_ratio : Coulomb ratio array (unused).

        Raises:
            - NotImplementedError : Always, as no dimensionless kernel is
            provided.
        """
        message = (
            "Dimensionless kernel not implemented in simple "
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
        """Calculate the dimensioned Brownian coagulation kernel.

        Leverages the `get_brownian_kernel_via_system_state` function to
        compute the kernel, which accounts for particle size, temperature,
        and pressure. The kernel typically has units of volume per time.

        Arguments:
            - particle : ParticleRepresentation containing the distribution
              and density or mass data needed for the kernel calculation.
            - temperature : System temperature in Kelvin.
            - pressure : System pressure in Pascals.

        Returns:
            - Brownian coagulation kernel values. Shape depends on the
              underlying distribution.

        Examples:
            ```py
            kernel_matrix = brownian_strat.kernel(particle_rep, 300, 101325)
            ```
        """
        return get_brownian_kernel_via_system_state(
            particle_radius=particle.get_radius(),
            particle_mass=particle.get_mass(),
            temperature=temperature,
            pressure=pressure,
        )
