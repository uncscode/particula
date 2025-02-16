"""Coagulation strategy module.
"""

from abc import ABC, abstractmethod
from typing import Union
import logging
from numpy.typing import NDArray
import numpy as np

from particula.particles.representation import ParticleRepresentation
from particula.dynamics.coagulation import rate
from particula.particles import properties
from particula.gas import properties as gas_properties

logger = logging.getLogger("particula")


class CoagulationStrategy(ABC):
    """
    Abstract class for defining a coagulation strategy. This class defines the
    methods that must be implemented by any coagulation strategy.

    Attributes:
        - distribution_type: The type of distribution to be used with the
            coagulation strategy. Default is "discrete", options are
            "discrete", "continuous_pdf", and "particle_resolved".

    Methods:
        kernel: Calculate the coagulation kernel.
        loss_rate: Calculate the coagulation loss rate.
        gain_rate: Calculate the coagulation gain rate.
        net_rate: Calculate the net coagulation rate.
        diffusive_knudsen: Calculate the diffusive Knudsen number.
        coulomb_potential_ratio: Calculate the Coulomb potential ratio.
    """

    def __init__(self, distribution_type: str):
        self.distribution_type = distribution_type

        if distribution_type not in [
            "discrete",
            "continuous_pdf",
            "particle_resolved",
        ]:
            raise ValueError(
                "Invalid distribution type. Must be one of 'discrete', "
                + "'continuous_pdf', or 'particle_resolved'."
            )

    @abstractmethod
    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Calculate the dimensionless coagulation kernel based on the particle
        properties interactions, diffusive Knudsen number and Coulomb
        potential.

        Args:
            - diffusive_knudsen : The diffusive Knudsen number
                for the particle [dimensionless].
            - coulomb_potential_ratio : The Coulomb potential
                ratio for the particle [dimensionless].

        Returns:
            The dimensionless coagulation kernel for the particle
                [dimensionless].
        """

    @abstractmethod
    def kernel(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the coagulation kernel based on the particle properties,
        temperature, and pressure.

        Args:
            particle: The particle for which the coagulation
                kernel is to be calculated.
            temperature: The temperature of the gas phase [K].
            pressure: The pressure of the gas phase [Pa].

        Returns:
            The coagulation kernel for the particle [m^3/s].
        """

    def loss_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the coagulation loss rate based on the particle radius,
        distribution, and the coagulation kernel.

        Args:
            particle: The particle for which the coagulation
                loss rate is to be calculated.
            kernel: The coagulation kernel.

        Returns:
            The coagulation loss rate for the particle [kg/s].

        Raises:
            ValueError : If the distribution type is not valid. Only
                'discrete' and 'continuous_pdf' are valid.
        """
        if self.distribution_type == "discrete":
            return rate.discrete_loss(
                concentration=particle.concentration,
                kernel=kernel,
            )
        if self.distribution_type == "continuous_pdf":
            return rate.continuous_loss(
                radius=particle.get_radius(),
                concentration=particle.concentration,
                kernel=kernel,
            )
        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete' or 'continuous_pdf'."
        )

    def gain_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the coagulation gain rate based on the particle radius,
        distribution, and the coagulation kernel. Used for discrete and
        continuous PDF distributions.

        Args:
            particle: The particle for which the coagulation
                gain rate is to be calculated.
            kernel: The coagulation kernel.

        Returns:
            The coagulation gain rate for the particle [kg/s].

        Raises:
            ValueError : If the distribution type is not valid. Only
                'discrete' and 'continuous_pdf' are valid.
        """
        if self.distribution_type == "discrete":
            return rate.discrete_gain(
                radius=particle.get_radius(),
                concentration=particle.concentration,
                kernel=kernel,
            )
        if self.distribution_type == "continuous_pdf":
            return rate.continuous_gain(
                radius=particle.get_radius(),
                concentration=particle.concentration,
                kernel=kernel,
            )
        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete' or 'continuous_pdf'."
        )

    def net_rate(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the net coagulation rate based on the particle radius,
        distribution, and the coagulation kernel.

        Arguments:
            - particle : The particle class for which the
                coagulation net rate is to be calculated.
            - temperature : The temperature of the gas phase [K].
            - pressure : The pressure of the gas phase [Pa].

        Returns:
            Union[float, NDArray[np.float64]]: The net coagulation rate for the
                particle [kg/s].
        """

        kernel = self.kernel(
            particle=particle, temperature=temperature, pressure=pressure
        )
        loss_rate = self.loss_rate(particle=particle, kernel=kernel)
        gain_rate = self.gain_rate(particle=particle, kernel=kernel)
        return gain_rate - loss_rate

    @abstractmethod
    def step(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> ParticleRepresentation:
        """
        Perform a single step of the coagulation process.

        Args:
            particle: The particle for which the coagulation step
                is to be performed.
            temperature: The temperature of the gas phase [K].
            pressure: The pressure of the gas phase [Pa].
            time_step: The time step for the coagulation process [s].

        Returns:
            ParticleRepresentation: The particle after the coagulation step.
        """

    def diffusive_knudsen(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """
        Calculate the diffusive Knudsen number based on the particle
        properties, temperature, and pressure.

        Args:
            particle: The particle for which the diffusive
                Knudsen number is to be calculated.
            temperature: The temperature of the gas phase [K].
            pressure: The pressure of the gas phase [Pa].

        Returns:
            NDArray[np.float64]: The diffusive Knudsen number for the particle
                [dimensionless].
        """
        # properties calculation for friction factor
        friction_factor = self.friction_factor(
            particle=particle, temperature=temperature, pressure=pressure
        )
        # coulomb potential ratio
        coulomb_potential_ratio = self.coulomb_potential_ratio(
            particle=particle, temperature=temperature
        )
        return properties.diffusive_knudsen_number(
            radius=particle.get_radius(),
            mass_particle=particle.get_mass(),
            friction_factor=friction_factor,
            coulomb_potential_ratio=coulomb_potential_ratio,
            temperature=temperature,
        )  # type: ignore

    def coulomb_potential_ratio(
        self, particle: ParticleRepresentation, temperature: float
    ) -> NDArray[np.float64]:
        """
        Calculate the Coulomb potential ratio based on the particle properties
        and temperature.

        Args:
            particle: The particles for which the Coulomb
                potential ratio is to be calculated.
            temperature: The temperature of the gas phase [K].

        Returns:
            The Coulomb potential ratio for the particle
                [dimensionless].
        """
        return properties.coulomb_enhancement.ratio(
            radius=particle.get_radius(),
            charge=particle.get_charge(),
            temperature=temperature,
        )  # type: ignore

    def friction_factor(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """
        Calculate the friction factor based on the particle properties,
        temperature, and pressure.

        Args:
            particle: The particle for which the friction factor
                is to be calculated.
            temperature: The temperature of the gas phase [K].
            pressure: The pressure of the gas phase [Pa].

        Returns:
            The friction factor for the particle [dimensionless].
        """
        dynamic_viscosity = gas_properties.get_dynamic_viscosity(
            temperature=temperature  # assume standard atmospheric composition
        )
        mean_free_path = gas_properties.molecule_mean_free_path(
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        knudsen_number = properties.calculate_knudsen_number(
            mean_free_path=mean_free_path,
            particle_radius=particle.get_radius(),
        )
        slip_correction = properties.cunningham_slip_correction(
            knudsen_number=knudsen_number
        )
        return properties.friction_factor(
            radius=particle.get_radius(),
            dynamic_viscosity=dynamic_viscosity,
            slip_correction=slip_correction,
        )  # type: ignore
