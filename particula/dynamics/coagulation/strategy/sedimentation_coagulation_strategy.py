"""
Sedimentation Coagulation Strategy
"""

from typing import Union
import logging
import numpy as np
from numpy.typing import NDArray

from particula.particles.representation import ParticleRepresentation
from particula.dynamics.coagulation.strategy.coagulation_strategy_abc import (
    CoagulationStrategyABC,
)
from particula.dynamics.coagulation.turbulent_shear_kernel import (
    get_turbulent_shear_kernel_st1956_via_system_state,
)

logger = logging.getLogger("particula")


class SedimentationCoagulationStrategy(CoagulationStrategyABC):
    """
    Sedimentation coagulation strategy.

    This implements the methods defined in `CoagulationStrategy` abstract
    class. Applied to the Seinfeld and Pandis (2016) sedimentation
    coagulation kernel.

    Arguments:
        - distribution_type : The type of distribution to be used with the
            coagulation strategy. Must be "discrete", "continuous_pdf", or
            "particle_resolved".
        - turbulent_kinetic_energy : The turbulent kinetic energy of the system
            [m^2/s^2].
        - fluid_density : The density of the fluid [kg/m^3].

    Methods:
    - kernel : Calculate the coagulation kernel.
    - loss_rate : Calculate the coagulation loss rate.
    - gain_rate : Calculate the coagulation gain rate.
    - net_rate : Calculate the net coagulation rate.

    References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
        physics, Chapter 13, Equation 13A.4.
    """

    def __init__(
        self,
        distribution_type: str,
        turbulent_kinetic_energy: float,
        fluid_density: float,
    ):
        CoagulationStrategyABC.__init__(
            self, distribution_type=distribution_type
        )
        self.turbulent_kinetic_energy = turbulent_kinetic_energy
        self.fluid_density = fluid_density

    def set_turbulent_kinetic_energy(self, turbulent_kinetic_energy: float):
        """Set the turbulent kinetic energy."""
        self.turbulent_kinetic_energy = turbulent_kinetic_energy
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
            turbulent_kinetic_energy=self.turbulent_kinetic_energy,
            temperature=temperature,
            fluid_density=self.fluid_density,
        )

    def step(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> ParticleRepresentation:

        particle.add_concentration(
            self.net_rate(  # type: ignore
                particle=particle, temperature=temperature, pressure=pressure
            )
            * time_step
        )
        return particle
