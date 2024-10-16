"""
Particle Vapor Equilibrium, condensation and evaporation
based on partial pressures to get dm/dt or other forms of
particle growth and decay.

From Seinfeld and Pandas: The Condensation (chapter 13) Equation 13.3
This function calculates the rate of change of the mass of an aerosol
particle with diameter Dp.

The rate of change of mass, dm, is given by the formula:
    dm/dt = 4π * radius * Di * Mi * f(Kn, alpha) * delta_pi / RT
where:
    radius is the radius of the particle,
    Di is the diffusion coefficient of species i,
    Mi is the molar mass of species i,
    f(Kn, alpha) is a transition function of the Knudsen number (Kn) and the
    mass accommodation coefficient (alpha),
    delta_pi is the partial pressure of species i in the gas phase vs the
    particle phase.
    R is the gas constant,
    T is the temperature.

    An additional denominator term is added to acount for temperature changes,
    which is need for cloud droplets, but can be used in general too.

This is also in Aerosol Modeling Chapter 2 Equation 2.40

Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and Physics:
From Air Pollution to Climate Change. In Wiley (3rd ed.).
John Wiley & Sons, Inc.

Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
(D. Topping & M. Bane, Eds.). Wiley. https://doi.org/10.1002/9781119625728

Units are all Base SI units.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
from numpy.typing import NDArray
import numpy as np

# particula imports
from particula.particles.representation import ParticleRepresentation
from particula.gas.species import GasSpecies
from particula.particles.properties import (
    calculate_knudsen_number,
    vapor_transition_correction,
    partial_pressure_delta,
)
from particula.gas.properties import molecule_mean_free_path
from particula.dynamics.condensation.mass_transfer import (
    first_order_mass_transport_k,
    mass_transfer_rate,
    calculate_mass_transfer,
)


# mass transfer abstract class
class CondensationStrategy(ABC):
    """
    Abstract base class for condensation strategies.

    This class defines the interface for various condensation models
    used in atmospheric physics. Subclasses should implement specific
    condensation algorithms based on different physical models and equations.

    Args:
        molar_mass: The molar mass of the species [kg/mol]. If a single value
            is provided, it will be used for all species.
        diffusion_coefficient: The diffusion coefficient of the species
            [m^2/s]. If a single value is provided, it will be used for all
            species. Default is 2e-5 m^2/s for air.
        accommodation_coefficient: The mass accommodation coefficient of the
            species. If a single value is provided, it will be used for all
            species. Default is 1.0.
    """

    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
    ):
        self.molar_mass = molar_mass
        self.diffusion_coefficient = diffusion_coefficient
        self.accommodation_coefficient = accommodation_coefficient
        self.update_gases = update_gases

    def mean_free_path(
        self,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the mean free path of the gas molecules based on the
        temperature, pressure, and dynamic viscosity of the gas.

        Args:
            temperature: The temperature of the gas [K].
            pressure: The pressure of the gas [Pa].
            dynamic_viscosity: The dynamic viscosity of the gas [Pa*s]. If not
            provided, it will be calculated based on the temperature

        Returns:
            Union[float, NDArray[np.float64]]: The mean free path of the gas
                molecules in meters (m).

        References:
            Mean Free Path:
            [Wikipedia](https://en.wikipedia.org/wiki/Mean_free_path)
        """
        return molecule_mean_free_path(
            molar_mass=self.molar_mass,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )

    def knudsen_number(
        self,
        radius: Union[float, NDArray[np.float64]],
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """The Knudsen number for a particle.

        Calculate the Knudsen number based on the mean free path of the gas
        molecules and the radius of the particle.

        Args:
            radius: The radius of the particle [m].
            temperature: The temperature of the gas [K].
            pressure: The pressure of the gas [Pa].
            dynamic_viscosity: The dynamic viscosity of the gas [Pa*s]. If
                not provided, it will be calculated based on the temperature

        Returns:
            The Knudsen number, which is the ratio of the mean free path to
                the particle radius.

        References:
            [Knudsen Number](https://en.wikipedia.org/wiki/Knudsen_number)
        """
        return calculate_knudsen_number(
            mean_free_path=self.mean_free_path(
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity,
            ),
            particle_radius=radius,
        )

    def first_order_mass_transport(
        self,
        radius: Union[float, NDArray[np.float64]],
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        """First-order mass transport coefficient per particle.

        Calculate the first-order mass transport coefficient, K, for a given
        particle based on the diffusion coefficient, radius, and vapor
        transition correction factor.

        Args:
            radius: The radius of the particle [m].
            temperature: The temperature at which the first-order mass
            transport coefficient is to be calculated.
            pressure: The pressure of the gas phase.
            dynamic_viscosity: The dynamic viscosity of the gas [Pa*s]. If not
            provided, it will be calculated based on the temperature

        Returns:
            The first-order mass transport coefficient per particle (m^3/s).

        References:
            - Aerosol Modeling, Chapter 2, Equation 2.49 (excluding particle
                number)
        """
        vapor_transition = vapor_transition_correction(
            knudsen_number=self.knudsen_number(
                radius=radius,
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=dynamic_viscosity,
            ),
            mass_accommodation=self.accommodation_coefficient,
        )
        return first_order_mass_transport_k(
            radius=radius,
            vapor_transition=vapor_transition,
            diffusion_coefficient=self.diffusion_coefficient,
        )

    @abstractmethod
    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        # pylint: disable=too-many-positional-arguments, too-many-arguments
        """Mass transfer rate for a particle.

        Calculate the mass transfer rate based on the difference in partial
        pressure and the first-order mass transport coefficient.

        Args:
            particle: The particle for which the mass transfer rate is to be
                calculated.
            gas_species: The gas species with which the particle is in contact.
            temperature: The temperature at which the mass transfer rate
                is to be calculated.
            pressure: The pressure of the gas phase.
            dynamic_viscosity: The dynamic viscosity of the gas [Pa*s]. If not
                provided, it will be calculated based on the temperature

        Returns:
            The mass transfer rate for the particle [kg/s].
        """

    @abstractmethod
    def rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """
        Calculate the rate of mass condensation for each particle due to
        each condensable gas species.

        The rate of condensation is determined based on the mass transfer rate,
        which is a function of particle properties, gas species properties,
        temperature, and pressure. This rate is then scaled by the
        concentration of particles in the system to get the overall
        condensation rate for each particle or bin.

        Args:
            particle (ParticleRepresentation): Representation of the particles,
                including properties such as size, concentration, and mass.
            gas_species (GasSpecies): The species of gas condensing onto the
                particles.
            temperature (float): The temperature of the system in Kelvin.
            pressure (float): The pressure of the system in Pascals.

        Returns:
            An array of condensation rates for each particle, scaled by
                particle concentration.
        """

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    @abstractmethod
    def step(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> Tuple[ParticleRepresentation, GasSpecies]:
        """
        Execute the condensation process for a given time step.

        Args:
            particle (ParticleRepresentation): The particle to modify.
            gas_species (GasSpecies): The gas species to condense onto the
                particle.
            temperature (float): The temperature of the system in Kelvin.
            pressure (float): The pressure of the system in Pascals.
            time_step (float): The time step for the process in seconds.

        Returns:
            (ParticleRepresentation, GasSpecies): The modified particle
                instance and the modified gas species instance.
        """


# Define a condensation strategy with no latent heat of vaporization effect
class CondensationIsothermal(CondensationStrategy):
    """Condensation strategy for isothermal conditions.

    Condensation strategy for isothermal conditions, where the temperature
    remains constant. This class implements the mass transfer rate calculation
    for condensation of particles based on partial pressures. No Latent heat
    of vaporization effect is considered.
    """

    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
    ):
        super().__init__(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion_coefficient,
            accommodation_coefficient=accommodation_coefficient,
            update_gases=update_gases,
        )

    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        # pylint: disable=too-many-positional-arguments, too-many-arguments

        # fill zero radius with a maximum value, the zero concentration
        # will ensure no mass transfer for those filled particles
        radius_with_fill = particle.get_radius()
        radius_with_fill = np.where(
            radius_with_fill == 0, np.max(radius_with_fill), radius_with_fill
        )
        # Calculate the first-order mass transport coefficient
        first_order_mass_transport = self.first_order_mass_transport(
            radius=radius_with_fill,
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        # calculate the partial pressure
        partial_pressure_particle = particle.activity.partial_pressure(
            pure_vapor_pressure=gas_species.get_pure_vapor_pressure(
                temperature
            ),
            mass_concentration=particle.get_species_mass(),
        )
        partial_pressure_gas = gas_species.get_partial_pressure(temperature)
        # calculate the kelvin term
        kelvin_term = particle.surface.kelvin_term(
            radius=radius_with_fill,
            molar_mass=self.molar_mass,
            mass_concentration=particle.get_species_mass(),
            temperature=temperature,
        )
        # calculate the pressure delta accounting for the kelvin term
        pressure_delta = partial_pressure_delta(
            partial_pressure_gas=partial_pressure_gas,
            partial_pressure_particle=partial_pressure_particle,
            kelvin_term=kelvin_term,
        )
        # Calculate the mass transfer rate per particle
        return mass_transfer_rate(
            pressure_delta=pressure_delta,
            first_order_mass_transport=first_order_mass_transport,
            temperature=temperature,
            molar_mass=self.molar_mass,
        )

    def rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:

        # Step 1: Calculate the mass transfer rate due to condensation
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )

        # Step 2: Reshape the particle concentration if necessary
        if mass_rate.ndim == 2:  # Multiple gas species  # type: ignore
            concentration = particle.concentration[:, np.newaxis]
        else:
            concentration = particle.concentration

        # Step 3: Calculate the overall condensation rate by scaling
        # mass rate by particle concentration
        rates = mass_rate * concentration

        return rates

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def step(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> Tuple[ParticleRepresentation, GasSpecies]:

        # Calculate the mass transfer rate
        mass_rate = self.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=temperature,
            pressure=pressure,
        )
        # calculate the mass gain or loss per bin
        mass_transfer = calculate_mass_transfer(
            mass_rate=mass_rate,  # type: ignore
            time_step=time_step,
            gas_mass=gas_species.get_concentration(),  # type: ignore
            particle_mass=particle.get_species_mass(),
            particle_concentration=particle.get_concentration(),
        )
        # apply the mass change
        particle.add_mass(added_mass=mass_transfer)
        if self.update_gases:
            # remove mass from gas phase concentration
            gas_species.add_concentration(
                added_concentration=-mass_transfer.sum(axis=0)
            )
        return particle, gas_species
