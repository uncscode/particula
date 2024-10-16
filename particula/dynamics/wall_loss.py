"""
Calculates the wall loss rate of a particle in a chamber with a given geometry.
"""

from typing import Union, Tuple
from numpy.typing import NDArray
import numpy as np

from particula.dynamics.properties.wall_loss_coefficient import (
    rectangle_wall_loss_coefficient_via_system_state,
    spherical_wall_loss_coefficient_via_system_state,
)


# pylint: disable=too-many-positional-arguments, too-many-arguments
def spherical_wall_loss_rate(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    particle_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_radius: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the wall loss rate of particles in a spherical chamber.

    This function computes the rate at which particles are lost to the walls
    of a spherical chamber, based on the system state. It uses the wall eddy
    diffusivity, particle properties (radius, density, concentration), and
    environmental conditions (temperature, pressure) to determine the loss
    rate.

    Arguments:
        wall_eddy_diffusivity: The rate of wall eddy diffusivity in inverse
            seconds (s⁻¹).
        particle_radius: The radius of the particle in meters (m).
        particle_density: The density of the particle in kilograms per cubic
            meter (kg/m³).
        particle_concentration: The concentration of particles in the chamber
            in particles per cubic meter (#/m³).
        temperature: The temperature of the system in Kelvin (K).
        pressure: The pressure of the system in Pascals (Pa).
        chamber_radius: The radius of the spherical chamber in meters (m).

    Returns:
        The wall loss rate of the particles in the chamber.
    """

    # Step 1: Calculate the wall loss coefficient
    loss_coefficient = spherical_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        particle_radius=particle_radius,
        particle_density=particle_density,
        temperature=temperature,
        pressure=pressure,
        chamber_radius=chamber_radius,
    )

    # Step 2: Calculate and return the wall loss rate
    return -loss_coefficient * particle_concentration


# pylint: disable=too-many-positional-arguments, too-many-arguments
def rectangle_wall_loss_rate(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    particle_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_dimensions: Tuple[float, float, float],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the wall loss rate of particles in a rectangular chamber.

    This function computes the rate at which particles are lost to the walls
    of a rectangular chamber, based on the system state. It uses the wall eddy
    diffusivity, particle properties (radius, density, concentration), and
    environmental conditions (temperature, pressure) to determine the loss
    rate. The chamber dimensions (length, width, height) are also taken
    into account.

    Arguments:
        wall_eddy_diffusivity: The rate of wall eddy diffusivity in inverse
            seconds (s⁻¹).
        particle_radius: The radius of the particle in meters (m).
        particle_density: The density of the particle in kilograms per cubic
            meter (kg/m³).
        particle_concentration: The concentration of particles in the chamber
            in particles per cubic meter (#/m³).
        temperature: The temperature of the system in Kelvin (K).
        pressure: The pressure of the system in Pascals (Pa).
        chamber_dimensions: A tuple containing the length, width, and height
            of the rectangular chamber in meters (m).

    Returns:
        The wall loss rate of the particles in the chamber.
    """

    # Step 1: Calculate the wall loss coefficient
    loss_coefficient = rectangle_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        particle_radius=particle_radius,
        particle_density=particle_density,
        temperature=temperature,
        pressure=pressure,
        chamber_dimensions=chamber_dimensions,
    )

    # Step 2: Calculate and return the wall loss rate
    return -loss_coefficient * particle_concentration
