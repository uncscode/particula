"""Functions and Method to calculate the wall loss rate for particles
in a chamber.

References:
    - Crump, J. G., & Seinfeld, J. H. (1981). TURBULENT DEPOSITION AND
        GRAVITATIONAL SEDIMENTATION OF AN AEROSOL IN A VESSEL OF ARBITRARY
        SHAPE. In J Aerosol Sct (Vol. 12, Issue 5).
        https://doi.org/10.1016/0021-8502(81)90036-7
    - Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
        loss rates in vessels. Aerosol Science and Technology, 2(3),
        303-309. https://doi.org/10.1080/02786828308958636
    - McMurry, P. H., & Rader, D. J. (1985). Aerosol Wall Losses in
        Electrically Charged Chambers. Aerosol Science and Technology,
        4(3), 249-268. https://doi.org/10.1080/02786828508959054
"""

from typing import Union, Tuple
import numpy as np
from numpy.typing import NDArray

from particula.particles.properties import (
    particle_diffusion_coefficient_via_system_state,
    particle_settling_velocity_via_system_state,
    debye_function,
)


def spherical_wall_loss_coefficient(
    wall_eddy_diffusivity: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]],
    settling_velocity: Union[float, NDArray[np.float64]],
    chamber_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the wall loss coefficient for a spherical chamber
    approximation.

    Arguments:
        wall_eddy_diffusivity: Rate of the wall eddy diffusivity.
        diffusion_coefficient: Diffusion coefficient of the
            particle.
        settling_velocity: Settling velocity of the particle.
        chamber_radius: Radius of the chamber.

    Returns:
        The calculated wall loss rate for a spherical chamber.

    References:
    - Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
        loss rates in vessels. Aerosol Science and Technology, 2(3),
        303-309. https://doi.org/10.1080/02786828308958636
    """
    debye_variable = (
        np.pi
        * settling_velocity
        / (2 * np.sqrt(wall_eddy_diffusivity * diffusion_coefficient))
    )
    return 6 * np.sqrt(wall_eddy_diffusivity * diffusion_coefficient) / (
        np.pi * chamber_radius
    ) * debye_function(variable=debye_variable) + settling_velocity / (
        4 * chamber_radius / 3
    )


def rectangle_wall_loss_coefficient(
    wall_eddy_diffusivity: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]],
    settling_velocity: Union[float, NDArray[np.float64]],
    chamber_dimensions: Tuple[float, float, float],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the wall loss coefficient, β₀, for a rectangular chamber.

    This function computes the wall loss coefficient for a rectangular-prism
    chamber, considering the wall eddy diffusivity, particle diffusion
    coefficient, and terminal settling velocity. The chamber dimensions
    (length, width, and height) are used to account for the geometry's impact
    on particle loss.

    Arguments:
        wall_eddy_diffusivity: Rate of wall diffusivity parameter in
            units of inverse seconds (s^-1).
        diffusion_coefficient: The particle diffusion coefficient
            in units of square meters per second (m²/s).
        settling_velocity: The terminal settling velocity of the
            particles, in units of meters per second (m/s).
        chamber_dimensions: A tuple of three floats representing the length
            (L), width (W), and height (H) of the rectangular chamber,
            in units of meters (m).

    Returns:
        The calculated wall loss rate (β₀) for the rectangular chamber.

    References:
        - Crump, J. G., & Seinfeld, J. H. (1981). TURBULENT DEPOSITION AND
            GRAVITATIONAL SEDIMENTATION OF AN AEROSOL IN A VESSEL OF ARBITRARY
            SHAPE. In J Aerosol Sct (Vol. 12, Issue 5).
            https://doi.org/10.1016/0021-8502(81)90036-7
    """
    length, width, height = chamber_dimensions  # Unpack the dimensions tuple

    # Using 1/tanh(x) for coth(x)
    coth_term = 1 / np.tanh(
        (np.pi * settling_velocity)
        / (4 * np.sqrt(wall_eddy_diffusivity * diffusion_coefficient))
    )

    # Calculate the wall loss coefficient
    return (length * width * height) ** -1 * (
        4
        * height
        * (length + width)
        * np.sqrt(wall_eddy_diffusivity * diffusion_coefficient)
        / np.pi
        + settling_velocity * length * width * coth_term
    )


# pylint: disable=too-many-positional-arguments, too-many-arguments
def spherical_wall_loss_coefficient_via_system_state(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the wall loss coefficient for a spherical chamber based on the
    system state.

    This function computes the wall loss coefficient for a spherical chamber
    using the system's physical state, including the wall eddy diffusivity,
    particle properties (radius, density), and environmental conditions
    (temperature, pressure). The chamber radius is also taken into account.

    Arguments:
        wall_eddy_diffusivity: The rate of wall eddy diffusivity in inverse
            seconds (s⁻¹).
        particle_radius: The radius of the particle in meters (m).
        particle_density: The density of the particle in kilograms per cubic
            meter (kg/m³).
        temperature: The temperature of the system in Kelvin (K).
        pressure: The pressure of the system in Pascals (Pa).
        chamber_radius: The radius of the spherical chamber in meters (m).

    Returns:
        The calculated wall loss coefficient for the spherical chamber.

    """

    # Step 1: Get particle diffusion coefficient
    diffusion_coefficient = particle_diffusion_coefficient_via_system_state(
        particle_radius=particle_radius,
        temperature=temperature,
        pressure=pressure,
    )

    # Step 2: Get particle settling velocity
    settling_velocity = particle_settling_velocity_via_system_state(
        particle_radius=particle_radius,
        particle_density=particle_density,
        temperature=temperature,
        pressure=pressure,
    )

    # Step 3: Calculate the wall loss coefficient for the spherical chamber
    return spherical_wall_loss_coefficient(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        diffusion_coefficient=diffusion_coefficient,
        settling_velocity=settling_velocity,
        chamber_radius=chamber_radius,
    )


# pylint: disable=too-many-positional-arguments, too-many-arguments
def rectangle_wall_loss_coefficient_via_system_state(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_dimensions: Tuple[float, float, float],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the wall loss coefficient for a rectangular chamber based on
    the system state.

    This function computes the wall loss coefficient for a rectangular chamber
    using the system's physical state, including the wall eddy diffusivity,
    particle properties (radius, density), and environmental conditions
    (temperature, pressure). The chamber dimensions (length, width, height)
    are also considered.

    Arguments:
        wall_eddy_diffusivity: The rate of wall eddy diffusivity in inverse
            seconds (s⁻¹).
        particle_radius: The radius of the particle in meters (m).
        particle_density: The density of the particle in kilograms per cubic
            meter (kg/m³).
        temperature: The temperature of the system in Kelvin (K).
        pressure: The pressure of the system in Pascals (Pa).
        chamber_dimensions: A tuple containing the length, width, and height
            of the rectangular chamber in meters (m).

    Returns:
        The calculated wall loss coefficient for the rectangular chamber.

    References:
        - Crump, J. G., & Seinfeld, J. H. (1981). TURBULENT DEPOSITION AND
            GRAVITATIONAL SEDIMENTATION OF AN AEROSOL IN A VESSEL OF ARBITRARY
            SHAPE. In J Aerosol Sct (Vol. 12, Issue 5).
            https://doi.org/10.1016/0021-8502(81)90036-7
    """
    # Step 1: Get particle settling velocity
    settling_velocity = particle_settling_velocity_via_system_state(
        particle_radius=particle_radius,
        particle_density=particle_density,
        temperature=temperature,
        pressure=pressure,
    )

    # Step 2: Get particle diffusion coefficient
    diffusion_coefficient = particle_diffusion_coefficient_via_system_state(
        particle_radius=particle_radius,
        temperature=temperature,
        pressure=pressure,
    )

    # Step 3: Calculate and return the wall loss coefficient for the chamber
    return rectangle_wall_loss_coefficient(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        diffusion_coefficient=diffusion_coefficient,
        settling_velocity=settling_velocity,
        chamber_dimensions=chamber_dimensions,
    )
