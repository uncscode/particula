"""Coagulation strategy module.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional
from numpy.typing import NDArray
import numpy as np

from particula.next.particles.representation import Particle
from particula.next.dynamics.coagulation import rate
from particula.next.dynamics.coagulation.brownian import (
    brownian_coagulation_kernel_via_system_state
)


class CoagulationStrategy(ABC):
    """
    Abstract class for defining a coagulation strategy. This class defines the
    methods that must be implemented by any coagulation strategy.

    Methods:
    --------
    - kernel (abstractmethod): Calculate the coagulation kernel.
    - loss_rate: Calculate the coagulation loss rate.
    - gain_rate: Calculate the coagulation gain rate.
    - net_rate: Calculate the net coagulation rate.
    """

    @abstractmethod
    def kernel(
        self,
        particle: Particle,
        temperature: float,
        pressure: float
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the coagulation kernel based on the particle properties,
        temperature, and pressure.

        Args:
        -----
        - particle (Particle class): The particle for which the coagulation
        kernel is to be calculated.
        - temperature (float): The temperature of the gas phase [K].
        - pressure (float): The pressure of the gas phase [Pa].

        Returns:
        --------
        - NDArray[np.float_]: The coagulation kernel for the particle [m^3/s].
        """

    def loss_rate(
        self,
        particle: Particle,
        kernel: NDArray[np.float_],
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the coagulation loss rate based on the particle radius,
        distribution, and the coagulation kernel.

        Args:
        -----
        - particle (Particle class): The particle for which the coagulation
        loss rate is to be calculated.
        - kernel (NDArray[np.float_]): The coagulation kernel.

        Returns:
        --------
        - Union[float, NDArray[np.float_]]: The coagulation loss rate for the
        particle [kg/s].

        Notes:
        ------
        May be abstracted to a separate module when different coagulation
        strategies are implemented (super droplet).
        """
        return rate.loss_rate(
            radius=particle.get_radius(),
            distribution=particle.distribution,
            kernel=kernel,
        )

    def gain_rate(
        self,
        particle: Particle,
        kernel: NDArray[np.float_],
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the coagulation gain rate based on the particle radius,
        distribution, and the coagulation kernel.

        Args:
        -----
        - particle (Particle class): The particle for which the coagulation
        gain rate is to be calculated.
        - kernel (NDArray[np.float_]): The coagulation kernel.

        Returns:
        --------
        - Union[float, NDArray[np.float_]]: The coagulation gain rate for the
        particle [kg/s].

        Notes:
        ------
        May be abstracted to a separate module when different coagulation
        strategies are implemented (super droplet).
        """
        return rate.gain_rate(
            radius=particle.get_radius(),
            distribution=particle.distribution,
            kernel=kernel,
        )

    def net_rate(
        self,
        particle: Particle,
        temperature: float,
        pressure: float
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the net coagulation rate based on the particle radius,
        distribution, and the coagulation kernel.

        Args:
        -----
        - particle (Particle class): The particle class for which the
        coagulation net rate is to be calculated.
        - temperature (float): The temperature of the gas phase [K].
        - pressure (float): The pressure of the gas phase [Pa].

        Returns:
        --------
        - Union[float, NDArray[np.float_]]: The net coagulation rate for the
        particle [kg/s].
        """
        kernel = self.kernel(particle, temperature, pressure)
        return self.gain_rate(particle, kernel) - self.loss_rate(
            particle, kernel)


# Define a coagulation strategy
class BrownianCoagulation(CoagulationStrategy):
    """
    Brownian coagulation strategy class. This class implements the methods
    defined in the CoagulationStrategy abstract class.

    Methods:
    --------
    - kernel: Calculate the coagulation kernel.
    - loss_rate: Calculate the coagulation loss rate.
    - gain_rate: Calculate the coagulation gain rate.
    - net_rate: Calculate the net coagulation rate.
    """

    def kernel(
        self,
        particle: Particle,
        temperature: float,
        pressure: float
    ) -> Union[float, NDArray[np.float_]]:

        return brownian_coagulation_kernel_via_system_state(
            radius_particle=particle.get_radius(),
            mass_particle=particle.get_mass(),
            temperature=temperature,
            pressure=pressure
        )
