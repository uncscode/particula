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
from particula.next.particles.representation import ParticleRepresentation
from particula.next.gas.species import GasSpecies
from particula.constants import GAS_CONSTANT  # type: ignore
from particula.next.particles.properties import (
    calculate_knudsen_number,
    vapor_transition_correction,
    partial_pressure_delta,
)
from particula.next.gas.properties import molecule_mean_free_path


def first_order_mass_transport_k(
    radius: Union[float, NDArray[np.float64]],
    vapor_transition: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]] = 2 * 1e-9,
) -> Union[float, NDArray[np.float64]]:
    """First-order mass transport coefficient per particle.

    Calculate the first-order mass transport coefficient, K, for a given radius
    diffusion coefficient, and vapor transition correction factor. For a
    single particle.

    Args:
        radius: The radius of the particle [m].
        diffusion_coefficient: The diffusion coefficient of the vapor [m^2/s],
        default to air.
        vapor_transition: The vapor transition correction factor. [unitless]

    Returns:
        Union[float, NDArray[np.float64]]: The first-order mass transport
        coefficient per particle (m^3/s).

    References:
        - Aerosol Modeling: Chapter 2, Equation 2.49 (excluding number)
        - Mass Diffusivity:
            [Wikipedia](https://en.wikipedia.org/wiki/Mass_diffusivity)
    """
    if (
        isinstance(vapor_transition, np.ndarray)
        and vapor_transition.dtype == np.float64
        and vapor_transition.ndim == 2
       ):  # extent radius
        radius = radius[:, np.newaxis]  # type: ignore
    return (
        4 * np.pi * radius
        * diffusion_coefficient * vapor_transition
    )  # type: ignore


def mass_transfer_rate(
    pressure_delta: Union[float, NDArray[np.float64]],
    first_order_mass_transport: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the mass transfer rate for a particle.

    Calculate the mass transfer rate based on the difference in partial
    pressure and the first-order mass transport coefficient.

    Args:
        pressure_delta: The difference in partial pressure between the gas
        phase and the particle phase.
        first_order_mass_transport: The first-order mass transport coefficient
        per particle.
        temperature: The temperature at which the mass transfer rate is to be
        calculated.

    Returns:
        Union[float, NDArray[np.float64]]: The mass transfer rate for the
        particle [kg/s].

    References:
        - Aerosol Modeling Chapter 2, Equation 2.41 (excluding particle number)
        - Seinfeld and Pandis: "Atmospheric Chemistry and Physics",
            Equation 13.3
    """
    return np.array(
        first_order_mass_transport
        * pressure_delta
        / (GAS_CONSTANT.m / molar_mass * temperature),
        dtype=np.float64,
    )


def calculate_mass_transfer(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Helper function that routes the mass transfer calculation to either the
    single-species or multi-species calculation functions based on the input
    dimensions of gas_mass.

    Args:
        mass_rate: The rate of mass transfer per particle (kg/s).
        time_step: The time step for the mass transfer calculation (seconds).
        gas_mass: The available mass of gas species (kg).
        particle_mass: The mass of each particle (kg).
        particle_concentration: The concentration of particles (number/m^3).

    Returns:
        The amount of mass transferred, accounting for gas and particle
            limitations.
    """
    if gas_mass.size == 1:  # Single species case
        return calculate_mass_transfer_single_species(
            mass_rate,
            time_step,
            gas_mass,
            particle_mass,
            particle_concentration,
        )
    # Multiple species case
    return calculate_mass_transfer_multiple_species(
        mass_rate,
        time_step,
        gas_mass,
        particle_mass,
        particle_concentration,
    )


def calculate_mass_transfer_single_species(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate mass transfer for a single gas species (m=1).

    Args:
        mass_rate: The rate of mass transfer per particle (number*kg/s).
        time_step: The time step for the mass transfer calculation (seconds).
        gas_mass: The available mass of gas species (kg).
        particle_mass: The mass of each particle (kg).
        particle_concentration: The concentration of particles (number/m^3).

    Returns:
        The amount of mass transferred for a single gas species.
    """
    # Step 1: Calculate the total mass to be transferred
    # (accounting for particle concentration)
    mass_to_change = mass_rate * time_step * particle_concentration
    # Step 2: Calculate the total requested mass
    total_requested_mass = mass_to_change.sum()
    # Step 3: Scale the mass if total requested mass exceeds available gas mass
    if total_requested_mass > gas_mass:
        scaling_factor = gas_mass / total_requested_mass
        mass_to_change *= scaling_factor
    # Step 4: Limit condensation by available gas mass
    condensible_mass_transfer = np.minimum(mass_to_change, gas_mass)
    # Step 5: Limit evaporation by available particle mass
    evaporative_mass_transfer = np.maximum(
        mass_to_change, -particle_mass * particle_concentration
    )
    # Step 6: Determine final transferable mass (condensation or evaporation)
    transferable_mass = np.where(
        mass_to_change > 0,  # Condensation scenario
        condensible_mass_transfer,  # Limited by gas mass
        evaporative_mass_transfer,  # Limited by particle mass
    )
    return transferable_mass


def calculate_mass_transfer_multiple_species(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate mass transfer for multiple gas species.

    Args:
        mass_rate: The rate of mass transfer per particle for each gas species
            (kg/s).
        time_step: The time step for the mass transfer calculation (seconds).
        gas_mass: The available mass of each gas species (kg).
        particle_mass: The mass of each particle for each gas species (kg).
        particle_concentration: The concentration of particles for each gas
            species (number/m^3).

    Returns:
        The amount of mass transferred for multiple gas species.
    """
    # Step 1: Calculate the total mass to change
    # (considering particle concentration)
    mass_to_change = (
        mass_rate * time_step * particle_concentration[:, np.newaxis]
    )

    # Step 2: Total requested mass for each gas species (sum over particles)
    total_requested_mass = mass_to_change.sum(axis=0)

    # Step 3: Create scaling factors where requested mass exceeds available
    # gas mass
    scaling_factors = np.ones_like(mass_to_change)
    scaling_mask = total_requested_mass > gas_mass

    # Apply scaling where needed (scaling along the gas species axis)
    scaling_factors[:, scaling_mask] = (
        gas_mass[scaling_mask] / total_requested_mass[scaling_mask]
    )

    # Step 4: Apply scaling factors to the mass_to_change
    mass_to_change *= scaling_factors

    # Step 5: Limit condensation by available gas mass
    condensible_mass_transfer = np.minimum(np.abs(mass_to_change), gas_mass)

    # Step 6: Limit evaporation by available particle mass
    evaporative_mass_transfer = np.maximum(
        mass_to_change, -particle_mass * particle_concentration[:, np.newaxis]
    )

    # Step 7: Determine the final transferable mass
    # (condensation or evaporation)
    transferable_mass = np.where(
        mass_to_change > 0,  # Condensation scenario
        condensible_mass_transfer,  # Limited by gas mass
        evaporative_mass_transfer,  # Limited by particle mass
    )

    return transferable_mass


# mass transfer abstract class
class CondensationStrategy(ABC):
    """Condensation strategy abstract class.

    Abstract class for mass transfer strategies, for condensation or
    evaporation of particles. This class should be subclassed to implement
    specific mass transfer strategies.

    Args:
        molar_mass: The molar mass of the species [kg/mol]. If a single value
        is provided, it will be used for all species.
        diffusion_coefficient: The diffusion coefficient of the species
        [m^2/s]. If a single value is provided, it will be used for all
        species. Default is 2*1e-9 m^2/s for air.
        accommodation_coefficient: The mass accommodation coefficient of the
        species. If a single value is provided, it will be used for all
        species. Default is 1.0.
    """

    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2 * 1e-9,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
    ):
        self.molar_mass = molar_mass
        self.diffusion_coefficient = diffusion_coefficient
        self.accommodation_coefficient = accommodation_coefficient

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
            Union[float, NDArray[np.float64]]: The Knudsen number, which is the
            ratio of the mean free path to the particle radius.

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
            Union[float, NDArray[np.float64]]: The first-order mass transport
            coefficient per particle (m^3/s).

        References:
            Aerosol Modeling, Chapter 2, Equation 2.49 (excluding particle
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
        # pylint: disable=too-many-arguments
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
            Union[float, NDArray[np.float64]]: The mass transfer rate for the
            particle [kg/s].
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
            An array of condensation rates for each particle,
            scaled by
            particle concentration.
        """

    # pylint: disable=too-many-arguments
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
            ParticleRepresentation: The modified particle instance.
            GasSpecies: The modified gas species instance.
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
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2 * 1e-9,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
    ):
        super().__init__(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion_coefficient,
            accommodation_coefficient=accommodation_coefficient,
        )

    def mass_transfer_rate(
        self,
        particle: ParticleRepresentation,
        gas_species: GasSpecies,
        temperature: float,
        pressure: float,
        dynamic_viscosity: Optional[float] = None,
    ) -> Union[float, NDArray[np.float64]]:
        # pylint: disable=too-many-arguments

        # Calculate the first-order mass transport coefficient
        first_order_mass_transport = self.first_order_mass_transport(
            radius=particle.get_radius(),
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
            radius=particle.get_radius(),
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

    # pylint: disable=too-many-arguments
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
        # remove mass from gas phase concentration
        gas_species.add_concentration(
            added_concentration=-mass_transfer.sum(axis=0)
        )
        return particle, gas_species
