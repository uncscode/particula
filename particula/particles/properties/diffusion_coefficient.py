"""Particle diffusion coefficient calculation."""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from particula.constants import BOLTZMANN_CONSTANT
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.gas.properties.mean_free_path import molecule_mean_free_path
from particula.particles.properties.aerodynamic_mobility_module import (
    particle_aerodynamic_mobility,
)
from particula.particles.properties.slip_correction_module import (
    cunningham_slip_correction,
)
from particula.particles.properties.knudsen_number_module import (
    calculate_knudsen_number,
)


def particle_diffusion_coefficient(
    temperature: Union[float, NDArray[np.float64]],
    aerodynamic_mobility: Union[float, NDArray[np.float64]],
    boltzmann_constant: float = BOLTZMANN_CONSTANT.m,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the diffusion coefficient of a particle.

    Arguments:
        temperature: The temperature at which the particle is
            diffusing, in Kelvin. Defaults to 298.15 K.
        boltzmann_constant: The Boltzmann constant. Defaults to the
            standard value of 1.380649 x 10^-23 J/K.
        aerodynamic_mobility: The aerodynamic mobility of
            the particle [m^2/s].

    Returns:
        The diffusion coefficient of the particle [m^2/s].
    """
    return boltzmann_constant * temperature * aerodynamic_mobility


def particle_diffusion_coefficient_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the diffusion coefficient of a particle.

    Arguments:
        temperature: The temperature of the system in Kelvin (K).
        particle_radius: The radius of the particle in meters (m).
        pressure: The pressure of the system in Pascals (Pa).

    Returns:
        The diffusion coefficient of the particle in square meters per
        second (m²/s).
    """

    # Step 1: Calculate gas properties
    _dynamic_viscosity = get_dynamic_viscosity(temperature=temperature)
    _mean_free_path = molecule_mean_free_path(
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=_dynamic_viscosity,
    )

    # Step 2: Particle properties in fluid
    _knudsen_number = calculate_knudsen_number(
        mean_free_path=_mean_free_path, particle_radius=particle_radius
    )
    _slip_correction_factor = cunningham_slip_correction(
        knudsen_number=_knudsen_number,
    )
    _aerodynamic_mobility = particle_aerodynamic_mobility(
        radius=particle_radius,
        slip_correction_factor=_slip_correction_factor,
        dynamic_viscosity=_dynamic_viscosity,
    )

    # Step 3: Calculate the particle diffusion coefficient
    return particle_diffusion_coefficient(
        temperature=temperature,
        aerodynamic_mobility=_aerodynamic_mobility,
    )
