"""Coagulation strategy module.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional
import logging
from numpy.typing import NDArray
import numpy as np

from particula.particles.representation import ParticleRepresentation
from particula.dynamics.coagulation import rate
from particula.dynamics.coagulation.brownian_kernel import (
    brownian_coagulation_kernel_via_system_state,
)
from particula.dynamics.coagulation.particle_resolved_method import (
    particle_resolved_coagulation_step,
)
from particula.dynamics.coagulation.kernel import KernelStrategy
from particula.particles import properties
from particula.gas import properties as gas_properties
from particula.util.reduced_quantity import reduced_self_broadcast

logger = logging.getLogger("particula")


class CoagulationStrategy(ABC):
    """
    Abstract class for defining a coagulation strategy. This class defines the
    methods that must be implemented by any coagulation strategy.

    Methods:
        kernel: Calculate the coagulation kernel.
        loss_rate: Calculate the coagulation loss rate.
        gain_rate: Calculate the coagulation gain rate.
        net_rate: Calculate the net coagulation rate.
        diffusive_knudsen: Calculate the diffusive Knudsen number.
        coulomb_potential_ratio: Calculate the Coulomb potential ratio.
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
            diffusive_knudsen: The diffusive Knudsen number
                for the particle [dimensionless].
            coulomb_potential_ratio: The Coulomb potential
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
            particle: The particle for which the coagulation
                loss rate is to be calculated.
            kernel: The coagulation kernel.

        Returns:
            The coagulation loss rate for the particle [kg/s].
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
            particle: The particle for which the coagulation
                gain rate is to be calculated.
            kernel: The coagulation kernel.

        Returns:
            The coagulation gain rate for the particle [kg/s].

        Notes:
            May be abstracted to a separate module when different coagulation
                strategies are implemented (super droplet).
        """

    @abstractmethod
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
            particle: The particle class for which the
                coagulation net rate is to be calculated.
            temperature: The temperature of the gas phase [K].
            pressure: The pressure of the gas phase [Pa].

        Returns:
            Union[float, NDArray[np.float64]]: The net coagulation rate for the
                particle [kg/s].
        """

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

    def net_rate(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:

        kernel = self.kernel(
            particle=particle, temperature=temperature, pressure=pressure
        )
        loss_rate = self.loss_rate(
            particle=particle, kernel=kernel  # type: ignore
        )
        gain_rate = self.gain_rate(
            particle=particle, kernel=kernel  # type: ignore
        )
        return gain_rate - loss_rate

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

    def net_rate(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:

        kernel = self.kernel(
            particle=particle, temperature=temperature, pressure=pressure
        )
        loss_rate = self.loss_rate(
            particle=particle, kernel=kernel  # type: ignore
        )
        gain_rate = self.gain_rate(
            particle=particle, kernel=kernel  # type: ignore
        )
        return gain_rate - loss_rate

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

    def net_rate(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:

        kernel = self.kernel(
            particle=particle, temperature=temperature, pressure=pressure
        )
        loss_rate = self.loss_rate(
            particle=particle, kernel=kernel  # type: ignore
        )
        gain_rate = self.gain_rate(
            particle=particle, kernel=kernel  # type: ignore
        )
        return gain_rate - loss_rate

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


class ParticleResolved(CoagulationStrategy):
    """
    Particle-resolved coagulation strategy class. This class implements the
    methods defined in the CoagulationStrategy abstract class. The kernel
    strategy is passed as an argument to the class, should use a dimensionless
    kernel representation.

    Methods:
        kernel: Calculate the coagulation kernel.
        loss_rate: Not implemented.
        gain_rate: Not implemented.
        net_rate: Not implemented.
        step: Perform a single step of the coagulation process.
    """

    def __init__(
        self,
        kernel_radius: Optional[NDArray[np.float64]] = None,
        kernel_bins_number: Optional[int] = None,
        kernel_bins_per_decade: int = 10,
    ):
        self.kernel_radius = kernel_radius
        self.kernel_bins_number = kernel_bins_number
        self.kernel_bins_per_decade = kernel_bins_per_decade

    def get_kernel_radius(
        self, particle: ParticleRepresentation
    ) -> NDArray[np.float64]:
        """Get the binning for the kernel radius.

        If the kernel radius is not set, it will be calculated based on the
        particle radius.

        Args:
            particle: The particle for which the kernel radius is to be
                calculated.

        Returns:
            The kernel radius for the particle [m].
        """
        if self.kernel_radius is not None:
            return self.kernel_radius
        # else find the non-zero min and max radii, the log space them
        min_radius = np.min(particle.get_radius()[particle.get_radius() > 0])
        max_radius = np.max(particle.get_radius()[particle.get_radius() > 0])
        if self.kernel_bins_number is not None:
            return np.logspace(
                np.log10(min_radius),
                np.log10(max_radius),
                num=self.kernel_bins_number,
                base=10,
                dtype=np.float64,
            )
        # else kernel bins per decade
        num = np.ceil(
            self.kernel_bins_per_decade * np.log10(max_radius / min_radius),
        )
        return np.logspace(
            np.log10(min_radius),
            np.log10(max_radius),
            num=int(num),
            base=10,
            dtype=np.float64,
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
        # need to update later with the correct mass dependency
        radius_bins = self.get_kernel_radius(particle)
        mass_bins = (
            4 / 3 * np.pi * np.power(radius_bins, 3) * 1000  # type: ignore
        )
        return brownian_coagulation_kernel_via_system_state(
            radius_particle=radius_bins,  # type: ignore
            mass_particle=mass_bins,  # type: ignore
            temperature=temperature,
            pressure=pressure,
        )

    def loss_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:

        message = "Loss rate not implemented in particle-resolved coagulation."
        logger.error(message)
        raise NotImplementedError(message)

    def gain_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:

        message = "Gain rate not implemented in particle-resolved coagulation."
        logger.error(message)
        raise NotImplementedError(message)

    def net_rate(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:

        message = "Net rate not implemented in particle-resolved coagulation."
        logger.error(message)
        raise NotImplementedError(message)

    def step(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> ParticleRepresentation:

        # need to add the particle resolved coagulation step
        loss_gain_indices = particle_resolved_coagulation_step(
            particle_radius=particle.get_radius(),
            kernel=self.kernel(  # type: ignore
                particle=particle, temperature=temperature, pressure=pressure
            ),
            kernel_radius=self.get_kernel_radius(particle),
            volume=particle.volume,
            time_step=time_step,
            random_generator=np.random.default_rng(),
        )
        particle.collide_pairs(loss_gain_indices)
        return particle
