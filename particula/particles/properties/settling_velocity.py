"""Particle settling velocity in a fluid."""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from particula.constants import STANDARD_GRAVITY
from particula.gas.properties.dynamic_viscosity import (
    get_dynamic_viscosity,
)
from particula.gas.properties.mean_free_path import (
    molecule_mean_free_path,
)
from particula.particles.properties.slip_correction_module import (
    cunningham_slip_correction,
)
from particula.particles.properties.knudsen_number_module import (
    calculate_knudsen_number,
)


def particle_settling_velocity(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
    gravitational_acceleration: float = STANDARD_GRAVITY.m,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the settling velocity of a particle in a fluid.

    Arguments:
        particle_radius: The radius of the particle [m].
        particle_density: The density of the particle [kg/m³].
        slip_correction_factor: The slip correction factor to
            account for non-continuum effects [dimensionless].
        gravitational_acceleration: The gravitational acceleration.
            Defaults to standard gravity [9.80665 m/s²].
        dynamic_viscosity: The dynamic viscosity of the fluid [Pa*s].

    Returns:
        The settling velocity of the particle in the fluid [m/s].

    """

    # Calculate the settling velocity using the given formula
    settling_velocity = (
        (2 * particle_radius) ** 2
        * particle_density
        * slip_correction_factor
        * gravitational_acceleration
        / (18 * dynamic_viscosity)
    )

    return settling_velocity


def particle_settling_velocity_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the settling velocity of a particle.

    Arguments:
        particle_radius: The radius of the particle in meters (m).
        particle_density: The density of the particle (kg/m³).
        temperature: The temperature of the system in Kelvin (K).
        pressure: The pressure of the system in Pascals (Pa).

    Returns:
        The settling velocity of the particle in meters per second (m/s).
    """

    # Step 1: Calculate the dynamic viscosity of the gas
    dynamic_viscosity = get_dynamic_viscosity(temperature=temperature)

    # Step 2: Calculate the mean free path of the gas molecules
    mean_free_path = molecule_mean_free_path(
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=dynamic_viscosity,
    )

    # Step 3: Calculate the Knudsen number (characterizes flow regime)
    knudsen_number = calculate_knudsen_number(
        mean_free_path=mean_free_path, particle_radius=particle_radius
    )

    # Step 4: Calculate the slip correction factor (Cunningham correction)
    slip_correction_factor = cunningham_slip_correction(
        knudsen_number=knudsen_number,
    )

    # Step 5: Calculate the particle settling velocity
    return particle_settling_velocity(
        particle_radius=particle_radius,
        particle_density=particle_density,
        slip_correction_factor=slip_correction_factor,
        dynamic_viscosity=dynamic_viscosity,
    )
