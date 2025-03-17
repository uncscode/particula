"""
Brownian coagulation strategy class.
"""

from typing import Union
import logging

import numpy as np
from numpy.typing import NDArray

from particula.particles.representation import ParticleRepresentation
from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state,
)

logger = logging.getLogger("particula")


class BrownianCoagulationStrategy(CoagulationStrategyABC):
    """
    Discrete Brownian coagulation strategy class. This class implements the
    methods defined in the CoagulationStrategy abstract class.

    Attributes:
        - distribution_type : The type of distribution to be used with the
            coagulation strategy. Options are "discrete", "continuous_pdf",
            and "particle_resolved".

    Methods:
    - kernel: Calculate the coagulation kernel.
    - loss_rate: Calculate the coagulation loss rate.
    - gain_rate: Calculate the coagulation gain rate.
    - net_rate: Calculate the net coagulation rate.

    References:
    - function `brownian_coagulation_kernel_via_system_state`
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
        physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
        Coefficient K12.
    """

    def __init__(self, distribution_type: str):
        CoagulationStrategyABC.__init__(
            self, distribution_type=distribution_type
        )

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
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

        return get_brownian_kernel_via_system_state(
            particle_radius=particle.get_radius(),
            particle_mass=particle.get_mass(),
            temperature=temperature,
            pressure=pressure,
        )
