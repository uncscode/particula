"""
Turbulent Shear Coagulation Strategy
"""

from typing import Union
import logging
import numpy as np
from numpy.typing import NDArray

from particula.particles.representation import ParticleRepresentation
from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from particula.dynamics.coagulation.turbulent_shear_kernel import (
    get_turbulent_shear_kernel_st1956_via_system_state,
)

logger = logging.getLogger("particula")


class TurbulentShearCoagulationStrategy(CoagulationStrategyABC):
    """
    Turbulent Shear coagulation strategy.

    This implements the methods defined in `CoagulationStrategy` abstract
    class. Applied to the Saffman and Turner (1956) turbulent shear
    coagulation kernel.

    Arguments:
        - distribution_type : The type of distribution to be used with the
            coagulation strategy. Must be "discrete", "continuous_pdf", or
            "particle_resolved".
        - turbulent_dissipation : The turbulent kinetic energy of the system
            [m^2/s^2].
        - fluid_density : The density of the fluid [kg/m^3].

    Methods:
    - kernel : Calculate the coagulation kernel.
    - loss_rate : Calculate the coagulation loss rate.
    - gain_rate : Calculate the coagulation gain rate.
    - net_rate : Calculate the net coagulation rate.

    References:
    - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops in
        turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
        https://doi.org/10.1017/S0022112056000020
    """

    def __init__(
        self,
        distribution_type: str,
        turbulent_dissipation: float,
        fluid_density: float,
    ):
        CoagulationStrategyABC.__init__(
            self, distribution_type=distribution_type
        )
        self.turbulent_dissipation = turbulent_dissipation
        self.fluid_density = fluid_density

    def set_turbulent_dissipation(self, turbulent_dissipation: float):
        """Set the turbulent kinetic energy."""
        self.turbulent_dissipation = turbulent_dissipation
        return self

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        message = (
            "Dimensionless kernel not implemented in turbulent shear "
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

        return get_turbulent_shear_kernel_st1956_via_system_state(
            particle_radius=particle.get_radius(),
            turbulent_dissipation=self.turbulent_dissipation,
            temperature=temperature,
            fluid_density=self.fluid_density,
        )
