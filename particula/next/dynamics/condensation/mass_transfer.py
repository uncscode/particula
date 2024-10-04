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

from typing import Union
from numpy.typing import NDArray
import numpy as np

# particula imports
from particula.constants import GAS_CONSTANT  # type: ignore


def first_order_mass_transport_k(
    radius: Union[float, NDArray[np.float64]],
    vapor_transition: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
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
        4 * np.pi * radius * diffusion_coefficient * vapor_transition
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
        The mass transfer rate for the particle [kg/s].

    References:
        - Aerosol Modeling Chapter 2, Equation 2.41 (excluding particle number)
        - Seinfeld and Pandis: "Atmospheric Chemistry and Physics",
            Equation 13.3
    """
    return np.array(
        first_order_mass_transport * pressure_delta * molar_mass / (
            GAS_CONSTANT.m * temperature),
        dtype=np.float64,
    )


def radius_transfer_rate(
    mass_rate: Union[float, NDArray[np.float64]],
    radius: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Convert mass rate to radius transfer rate.

    Convert the mass rate to a radius transfer rate based on the
    volume of the particle.

    Args:
        mass_rate: The mass transfer rate for the particle [kg/s].
        radius: The radius of the particle [m].
        density: The density of the particle [kg/m^3].

    Returns:
        The radius growth rate for the particle [m/s].
    """
    if isinstance(mass_rate, np.ndarray) and mass_rate.ndim == 2:
        radius = radius[:, np.newaxis]  # type: ignore
    return mass_rate / (density * 4 * np.pi * radius ** 2)  # type: ignore


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
