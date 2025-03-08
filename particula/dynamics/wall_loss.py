"""
Calculates the wall loss rate of a particle in a chamber with a given geometry.
"""

from typing import Union, Tuple
from numpy.typing import NDArray
import numpy as np

from particula.dynamics.properties.wall_loss_coefficient import (
    get_rectangle_wall_loss_coefficient_via_system_state,
    get_spherical_wall_loss_coefficient_via_system_state,
)


# pylint: disable=too-many-positional-arguments, too-many-arguments
def get_spherical_wall_loss_rate(
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

    This function calculates the rate at which particles deposit onto the
    walls of a spherical chamber. The calculation is based on the wall eddy
    diffusivity and key particle properties (radius, density, concentration),
    together with environmental conditions (temperature, pressure).
    The loss rate is determined via:

    - L = -(k × c)
        - L is the wall loss rate [#/m³·s],
        - k is the wall loss coefficient [1/s] from the system state,
        - c is the particle concentration [#/m³].

    Arguments:
        - wall_eddy_diffusivity : Wall eddy diffusivity in s⁻¹.
        - particle_radius : Particle radius in m.
        - particle_density : Particle density in kg/m³.
        - particle_concentration : Particle concentration in #/m³.
        - temperature : System temperature in K.
        - pressure : System pressure in Pa.
        - chamber_radius : Radius of the spherical chamber in m.

    Returns:
        - The wall loss rate (float or NDArray[np.float64]) in #/m³·s.

    Examples:
        ```py title="Example"
        import particula as par
        rate = par.dynamics.wall_loss.get_spherical_wall_loss_rate(
            wall_eddy_diffusivity=1e-3,
            particle_radius=1e-7,
            particle_density=1000,
            particle_concentration=1e11,
            temperature=298,
            pressure=101325,
            chamber_radius=0.5
        )
        print(rate)
        # Example output: -1.2e8
        ```

    References:
        - Wikipedia contributors, "Aerosol dynamics," Wikipedia,
          https://en.wikipedia.org/wiki/Aerosol.
    """

    # Step 1: Calculate the wall loss coefficient
    loss_coefficient = get_spherical_wall_loss_coefficient_via_system_state(
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
def get_rectangle_wall_loss_rate(
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

    This function calculates the rate of particle deposition onto the walls
    of a rectangular chamber, given the wall eddy diffusivity, particle
    properties (radius, density, concentration), and environmental conditions
    (temperature, pressure). The final loss rate is computed via:

    - L = -(k × c)
        - L is the wall loss rate [#/m³·s],
        - k is the wall loss coefficient [1/s],
        - c is the particle concentration [#/m³].

    Arguments:
        - wall_eddy_diffusivity : Wall eddy diffusivity in s⁻¹.
        - particle_radius : Particle radius in m.
        - particle_density : Particle density in kg/m³.
        - particle_concentration : Particle concentration in #/m³.
        - temperature : System temperature in K.
        - pressure : System pressure in Pa.
        - chamber_dimensions : (length, width, height) of the
            rectangular chamber in m.

    Returns:
        - The wall loss rate (float or NDArray[np.float64]) in #/m³·s.

    Examples:
        ```py title="Example"
        import particula as par
        loss_rate = par.dynamics.wall_loss.get_rectangle_wall_loss_rate(
            wall_eddy_diffusivity=1e-4,
            particle_radius=5e-8,
            particle_density=1200,
            particle_concentration=2e10,
            temperature=300,
            pressure=101325,
            chamber_dimensions=(1.0, 0.5, 0.5)
        )
        print(loss_rate)
        # Example output: -4.6e7
        ```

    References:
        - J. Hinds, "Aerosol Technology," 2nd ed., John Wiley & Sons, 1999.
    """

    # Step 1: Calculate the wall loss coefficient
    loss_coefficient = get_rectangle_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        particle_radius=particle_radius,
        particle_density=particle_density,
        temperature=temperature,
        pressure=pressure,
        chamber_dimensions=chamber_dimensions,
    )

    # Step 2: Calculate and return the wall loss rate
    return -loss_coefficient * particle_concentration
