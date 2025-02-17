"""
Turbulent DNS coagulation strategy, for larger particles greater than
1 microns.
"""

from typing import Union
import logging
import numpy as np
from numpy.typing import NDArray

from particula.particles.representation import ParticleRepresentation
from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.turbulent_dns_kernel_ao2008 import (
    get_turbulent_dns_kernel_ao2008_via_system_state,
)

logger = logging.getLogger("particula")


class TurbulentDNSCoagulationStrategy(CoagulationStrategyABC):
    """
    Turbulent DNS coagulation strategy.

    This implements the methods defined in `CoagulationStrategy` abstract
    class. Applied with the turbulent DNS kernel from Ayala et al. (2008).

    Arguments:
        - distribution_type : The type of distribution to be used with the
            coagulation strategy. Must be "discrete", "continuous_pdf", or
            "particle_resolved".
        - turbulent_dissipation : The turbulent kinetic energy of the system
            [m^2/s^2]. DNS fits are for 0.001, 0.01, and 0.04 [m^2/s^2].
        - fluid_density : The density of the fluid [kg/m^3].
        - reynolds_lambda : The Reynolds lambda of air, DNS fits are for
            23 and 74 [dimensionless].
        - relative_velocity : The relative velocity of the air [m/s].

    Methods:
    - kernel : Calculate the coagulation kernel.
    - loss_rate : Calculate the coagulation loss rate.
    - gain_rate : Calculate the coagulation gain rate.
    - net_rate : Calculate the net coagulation rate.

    References:
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """

    def __init__(
        self,
        distribution_type: str,
        turbulent_dissipation: float,
        fluid_density: float,
        reynolds_lambda: float,
        relative_velocity: float,
    ):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        CoagulationStrategyABC.__init__(
            self, distribution_type=distribution_type
        )
        self.turbulent_dissipation = turbulent_dissipation
        self.fluid_density = fluid_density
        self.reynolds_lambda = reynolds_lambda
        self.relative_velocity = relative_velocity

    def set_turbulent_dissipation(self, turbulent_dissipation: float):
        """Set the turbulent kinetic energy."""
        self.turbulent_dissipation = turbulent_dissipation
        return self

    def set_reynolds_lambda(self, reynolds_lambda: float):
        """Set the Reynolds lambda."""
        self.reynolds_lambda = reynolds_lambda
        return self

    def set_relative_velocity(self, relative_velocity: float):
        """Set the relative velocity."""
        self.relative_velocity = relative_velocity
        return self

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        message = (
            "Dimensionless kernel not implemented in turbulent DNS "
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

        return get_turbulent_dns_kernel_ao2008_via_system_state(
            particle_radius=particle.get_radius(),
            particle_density=particle.get_density(),
            fluid_density=self.fluid_density,
            turbulent_dissipation=self.turbulent_dissipation,
            re_lambda=self.reynolds_lambda,
            relative_velocity=self.relative_velocity,
            temperature=temperature,
        )
