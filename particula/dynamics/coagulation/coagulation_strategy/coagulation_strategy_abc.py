"""Coagulation strategy module."""

from abc import ABC, abstractmethod
from typing import Union, Optional
import logging
from numpy.typing import NDArray
import numpy as np

from particula.particles.representation import ParticleRepresentation
from particula.dynamics.coagulation import coagulation_rate
from particula.particles import properties
from particula.gas import properties as gas_properties
from particula.particles.change_particle_representation import (
    get_particle_resolved_binned_radius,
    get_speciated_mass_representation_from_particle_resolved,
)
from particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method import (
    get_particle_resolved_coagulation_step,
)

logger = logging.getLogger("particula")


class CoagulationStrategyABC(ABC):
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

    def __init__(
        self,
        distribution_type: str,
        particle_resolved_kernel_radius: Optional[NDArray[np.float64]] = None,
        particle_resolved_kernel_bins_number: Optional[int] = None,
        particle_resolved_kernel_bins_per_decade: int = 10,
    ):
        if distribution_type not in [
            "discrete",
            "continuous_pdf",
            "particle_resolved",
        ]:
            raise ValueError(
                "Invalid distribution type. Must be one of 'discrete', "
                + "'continuous_pdf', or 'particle_resolved'."
            )
        self.distribution_type = distribution_type
        self.random_generator = np.random.default_rng()
        # for particle resolved coagulation strategy
        self.particle_resolved_radius = particle_resolved_kernel_radius
        self.particle_resolved_bins_number = (
            particle_resolved_kernel_bins_number
        )
        self.particle_resolved_bins_per_decade = (
            particle_resolved_kernel_bins_per_decade
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
            return coagulation_rate.get_coagulation_loss_rate_discrete(
                concentration=particle.concentration,
                kernel=kernel,
            )
        if self.distribution_type == "continuous_pdf":
            return coagulation_rate.get_coagulation_loss_rate_continuous(
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
            return coagulation_rate.get_coagulation_gain_rate_discrete(
                radius=particle.get_radius(),
                concentration=particle.concentration,
                kernel=kernel,
            )
        if self.distribution_type == "continuous_pdf":
            return coagulation_rate.get_coagulation_gain_rate_continuous(
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

        if self.distribution_type in ["discrete", "continuous_pdf"]:
            particle.add_concentration(
                self.net_rate(  # type: ignore
                    particle=particle,
                    temperature=temperature,
                    pressure=pressure,
                )
                * time_step
            )
            return particle

        if self.distribution_type == "particle_resolved":
            # get the kernel radius
            kernel_radius = get_particle_resolved_binned_radius(
                particle=particle,
                bin_radius=self.particle_resolved_radius,
                total_bins=self.particle_resolved_bins_number,
                bins_per_radius_decade=self.particle_resolved_bins_per_decade,
            )
            # convert particle representation to calculate kernel
            kernel_particle = (
                get_speciated_mass_representation_from_particle_resolved(
                    particle=particle,
                    bin_radius=kernel_radius,
                )
            )
            # calculate step
            loss_gain_indices = get_particle_resolved_coagulation_step(
                particle_radius=particle.get_radius(),
                kernel=self.kernel(
                    particle=kernel_particle,
                    temperature=temperature,
                    pressure=pressure,
                ),
                kernel_radius=kernel_radius,
                volume=particle.get_volume(),
                time_step=time_step,
                random_generator=self.random_generator,
            )
            return particle.collide_pairs(loss_gain_indices)

        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete', 'continuous_pdf', or"
            " 'particle_resolved'."
        )

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
        return properties.get_diffusive_knudsen_number(
            particle_radius=particle.get_radius(),
            particle_mass=particle.get_mass(),
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
        return properties.coulomb_enhancement.get_coulomb_enhancement_ratio(
            particle_radius=particle.get_radius(),
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
        mean_free_path = gas_properties.get_molecule_mean_free_path(
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
        return properties.get_friction_factor(
            particle_radius=particle.get_radius(),
            dynamic_viscosity=dynamic_viscosity,
            slip_correction=slip_correction,
        )  # type: ignore
