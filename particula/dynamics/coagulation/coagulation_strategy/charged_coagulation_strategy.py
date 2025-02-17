"""
Charged particle coagulation strategy.
"""

from typing import Union
import logging

import numpy as np
from numpy.typing import NDArray

from particula.particles.representation import ParticleRepresentation
from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from particula.util.reduced_quantity import reduced_self_broadcast
from particula.dynamics.coagulation.charged_kernel_strategy import (
    KernelStrategy,
)

logger = logging.getLogger("particula")


class ChargedCoagulationStrategy(CoagulationStrategyABC):
    """
    General charged dependent brownian coagulation strategy.

    This class implements the methods defined in the CoagulationStrategy
    abstract class. The kernel strategy is passed as an argument to the class,
    to use a dimensionless kernel representation.

    Parameters:
        - kernel_strategy : The kernel strategy to be used for the coagulation,
            from the KernelStrategy class.

    Methods:
    - kernel: Calculate the coagulation kernel.
    - loss_rate: Calculate the coagulation loss rate.
    - gain_rate: Calculate the coagulation gain rate.
    - net_rate: Calculate the net coagulation rate.
    """

    def __init__(
        self, distribution_type: str, kernel_strategy: KernelStrategy
    ):
        CoagulationStrategyABC.__init__(
            self, distribution_type=distribution_type
        )
        self.kernel_strategy = kernel_strategy

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.kernel_strategy.dimensionless(
            diffusive_knudsen=diffusive_knudsen,
            coulomb_potential_ratio=coulomb_potential_ratio,
        )

    def kernel(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        diffusive_knudsen = self.diffusive_knudsen(
            particle=particle, temperature=temperature, pressure=pressure
        )
        coulomb_potential_ratio = self.coulomb_potential_ratio(
            particle=particle, temperature=temperature
        )
        dimensionless_kernel = self.dimensionless_kernel(
            diffusive_knudsen=diffusive_knudsen,
            coulomb_potential_ratio=coulomb_potential_ratio,
        )
        friction_factor = self.friction_factor(
            particle=particle, temperature=temperature, pressure=pressure
        )
        # Calculate the pairwise sum of radii
        radius = particle.get_radius()
        sum_of_radii = radius[:, np.newaxis] + radius
        # square matrix of mass
        reduced_mass = reduced_self_broadcast(particle.get_mass())
        # square matrix of friction factor
        reduced_friction_factor = reduced_self_broadcast(friction_factor)

        return self.kernel_strategy.kernel(
            dimensionless_kernel=dimensionless_kernel,
            coulomb_potential_ratio=coulomb_potential_ratio,
            sum_of_radii=sum_of_radii,
            reduced_mass=reduced_mass,
            reduced_friction_factor=reduced_friction_factor,
        )
