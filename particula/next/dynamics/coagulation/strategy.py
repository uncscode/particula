"""Coagulation strategy module.
"""

from abc import ABC, abstractmethod
from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.next.particles.representation import ParticleRepresentation
from particula.next.dynamics.coagulation import rate
from particula.next.dynamics.coagulation.brownian_kernel import (
    brownian_coagulation_kernel_via_system_state,
)
from particula.next.dynamics.coagulation.kernel import KernelStrategy
from particula.next.particles import properties
from particula.next.gas import properties as gas_properties
from particula.util.reduced_quantity import reduced_self_broadcast


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
    - diffusive_knudsen: Calculate the diffusive Knudsen number.
    - coulomb_potential_ratio: Calculate the Coulomb potential ratio.
    """

    @abstractmethod
    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Calculate the dimensionless coagulation kernel based on the particle
        properties interactions,
        diffusive Knudsen number and Coulomb potential

        Args:
        -----
        - diffusive_knudsen (NDArray[np.float64]): The diffusive Knudsen number
        for the particle [dimensionless].
        - coulomb_potential_ratio (NDArray[np.float64]): The Coulomb potential
        ratio for the particle [dimensionless].

        Returns:
        --------
        - NDArray[np.float64]: The dimensionless coagulation kernel for the
        particle [dimensionless].
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
        -----
        - particle (Particle class): The particle for which the coagulation
        kernel is to be calculated.
        - temperature (float): The temperature of the gas phase [K].
        - pressure (float): The pressure of the gas phase [Pa].

        Returns:
        --------
        - NDArray[np.float64]: The coagulation kernel for the particle [m^3/s].
        """

    @abstractmethod
    def loss_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the coagulation loss rate based on the particle radius,
        distribution, and the coagulation kernel.

        Args:
        -----
        - particle (Particle class): The particle for which the coagulation
        loss rate is to be calculated.
        - kernel (NDArray[np.float64]): The coagulation kernel.

        Returns:
        --------
        - Union[float, NDArray[np.float64]]: The coagulation loss rate for the
        particle [kg/s].
        """

    @abstractmethod
    def gain_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the coagulation gain rate based on the particle radius,
        distribution, and the coagulation kernel.

        Args:
        -----
        - particle (Particle class): The particle for which the coagulation
        gain rate is to be calculated.
        - kernel (NDArray[np.float64]): The coagulation kernel.

        Returns:
        --------
        - Union[float, NDArray[np.float64]]: The coagulation gain rate for the
        particle [kg/s].

        Notes:
        ------
        May be abstracted to a separate module when different coagulation
        strategies are implemented (super droplet).
        """

    def net_rate(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
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
        - Union[float, NDArray[np.float64]]: The net coagulation rate for the
        particle [kg/s].
        """
        kernel = self.kernel(particle, temperature, pressure)
        return self.gain_rate(
            particle, kernel
        ) - self.loss_rate(  # type: ignore
            particle, kernel
        )  # type: ignore

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
        -----
        - particle (Particle class): The particle for which the diffusive
        Knudsen number is to be calculated.
        - temperature (float): The temperature of the gas phase [K].
        - pressure (float): The pressure of the gas phase [Pa].

        Returns:
        --------
        - NDArray[np.float64]: The diffusive Knudsen number for the particle
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
        -----
        - particle (Particle class): The particles for which the Coulomb
        potential ratio is to be calculated.
        - temperature (float): The temperature of the gas phase [K].

        Returns:
        --------
        - NDArray[np.float64]: The Coulomb potential ratio for the particle
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
        -----
        - particle (Particle class): The particle for which the friction factor
        is to be calculated.
        - temperature (float): The temperature of the gas phase [K].
        - pressure (float): The pressure of the gas phase [Pa].

        Returns:
        --------
        - NDArray[np.float64]: The friction factor for the particle
        [dimensionless].
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


# Define a coagulation strategy
class DiscreteSimple(CoagulationStrategy):
    """
    Discrete Brownian coagulation strategy class. This class implements the
    methods defined in the CoagulationStrategy abstract class.

    Methods:
    --------
    - kernel: Calculate the coagulation kernel.
    - loss_rate: Calculate the coagulation loss rate.
    - gain_rate: Calculate the coagulation gain rate.
    - net_rate: Calculate the net coagulation rate.
    """

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:

        raise NotImplementedError(
            "Dimensionless kernel not implemented in \
            simple coagulation strategy. Use a general strategy."
        )

    def kernel(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:

        return brownian_coagulation_kernel_via_system_state(
            radius_particle=particle.get_radius(),
            mass_particle=particle.get_mass(),
            temperature=temperature,
            pressure=pressure,
        )

    def loss_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:

        return rate.discrete_loss(
            concentration=particle.concentration,
            kernel=kernel,
        )

    def gain_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:

        return rate.discrete_gain(
            radius=particle.get_radius(),
            concentration=particle.concentration,
            kernel=kernel,
        )


class DiscreteGeneral(CoagulationStrategy):
    """
    Discrete general coagulation strategy class. This class implements the
    methods defined in the CoagulationStrategy abstract class. The kernel
    strategy is passed as an argument to the class, to use a dimensionless
    kernel representation.

    Attributes:
    -----------
    - kernel_strategy: The kernel strategy to be used for the coagulation, from
    the KernelStrategy class.

    Methods:
    --------
    - kernel: Calculate the coagulation kernel.
    - loss_rate: Calculate the coagulation loss rate.
    - gain_rate: Calculate the coagulation gain rate.
    - net_rate: Calculate the net coagulation rate.
    """

    def __init__(self, kernel_strategy: KernelStrategy):
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

    def loss_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:

        return rate.discrete_loss(
            concentration=particle.concentration,
            kernel=kernel,
        )

    def gain_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:

        return rate.discrete_gain(
            radius=particle.get_radius(),
            concentration=particle.concentration,
            kernel=kernel,
        )


class ContinuousGeneralPDF(CoagulationStrategy):
    """
    Continuous PDF coagulation strategy class. This class implements the
    methods defined in the CoagulationStrategy abstract class. The kernel
    strategy is passed as an argument to the class, should use a dimensionless
    kernel representation.

    Methods:
        kernel: Calculate the coagulation kernel.
        loss_rate: Calculate the coagulation loss rate.
        gain_rate: Calculate the coagulation gain rate.
        net_rate: Calculate the net coagulation rate.
    """

    def __init__(self, kernel_strategy: KernelStrategy):
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

    def loss_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:

        return rate.continuous_loss(
            radius=particle.get_radius(),
            concentration=particle.concentration,
            kernel=kernel,
        )

    def gain_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:

        return rate.continuous_gain(
            radius=particle.get_radius(),
            concentration=particle.concentration,
            kernel=kernel,
        )
