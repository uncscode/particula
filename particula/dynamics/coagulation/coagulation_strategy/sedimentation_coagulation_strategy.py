"""Sedimentation coagulation strategies for aerosols.

Provides a sedimentation-based collision kernel following Seinfeld & Pandis
(2016). Implements a strategy class for gravitational-settling-driven
particle collisions.
"""

import logging
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.particles.representation import ParticleRepresentation

from ..sedimentation_kernel import (
    get_sedimentation_kernel_sp2016_via_system_state as get_kernel_sp2016,
)
from .coagulation_strategy_abc import CoagulationStrategyABC

logger = logging.getLogger("particula")


class SedimentationCoagulationStrategy(CoagulationStrategyABC):
    """Sedimentation coagulation strategy for aerosol particles.

    Implements the Seinfeld & Pandis (2016) sedimentation kernel as part of
    the CoagulationStrategyABC. This approach models collisions driven by
    gravitational settling.

    Attributes:
        distribution_type : The particle distribution type ("discrete",
        "continuous_pdf", or "particle_resolved").

    Methods:
    - dimensionless_kernel : Raises NotImplementedError for this strategy.
    - kernel : Return the sedimentation coagulation kernel [m^3/s].
    - loss_rate : (Inherited) Calculate coagulation loss rate.
    - gain_rate : (Inherited) Calculate coagulation gain rate.
    - net_rate : (Inherited) Calculate net coagulation rate.
    - step : Perform a single step of coagulation.
    - diffusive_knudsen : Calculate the diffusive Knudsen number.
    - coulomb_potential_ratio : Compute Coulomb potential ratio.
    - friction_factor : Compute the effective friction factor.

    Examples:
        ```py title="Sedimentation Coagulation Strategy Example"
        import particula as par
        strategy = SedimentationCoagulationStrategy(
            distribution_type="discrete"
        )
        # Use strategy.kernel(aerosol_particle, 298.15, 101325) to get the
        # sedimentation kernel.
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
          Physics, Chapter 13, Equation 13A.4, Wiley.
    """

    def __init__(self, distribution_type: str):
        """Initialize the sedimentation coagulation strategy.

        Arguments:
            distribution_type: Type of particle distribution ("discrete",
                "continuous_pdf", or "particle_resolved").
        """
        super().__init__(distribution_type=distribution_type)

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Raise NotImplementedError for dimensionless kernel in sedimentation
        strategy.

        This method is not applicable to sedimentation-based collisions.

        Arguments:
            - diffusive_knudsen : The diffusive Knudsen number [dimensionless].
            - coulomb_potential_ratio : The Coulomb potential ratio
              [dimensionless].

        Raises:
            - NotImplementedError : Always raised for this strategy.
        """
        message = (
            "Dimensionless kernel not implemented in sedimentation "
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
        """Compute the sedimentation coagulation kernel [m^3/s].

        Uses the Seinfeld & Pandis (2016) sedimentation kernel via
        `get_sedimentation_kernel_sp2016_via_system_state`.

        Arguments:
            - particle : The ParticleRepresentation providing particle radius
              and density.
            - temperature : The system temperature [K].
            - pressure : The system pressure [Pa].

        Returns:
            - The sedimentation coagulation kernel.

        Examples:
            ```py
            k_values = strategy.kernel(
                ParticleRepresentation, temperature=298.15, pressure=101325
            )
            # k_values may be a single float or an array
            ```
        """
        return get_kernel_sp2016(
            particle_radius=particle.get_radius(),
            particle_density=particle.get_mean_effective_density(),
            temperature=temperature,
            pressure=pressure,
            calculate_collision_efficiency=False,
        )
