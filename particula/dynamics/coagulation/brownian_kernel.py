"""
The basic Brownian coagulation kernel for aerosol particles, as described by
Seinfeld and Pandis (2016) from Fuchs' theory, Chapter 13 table 13.1.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.constants import BOLTZMANN_CONSTANT
from particula.particles import properties
from particula.gas.properties import (
    get_dynamic_viscosity,
    get_molecule_mean_free_path,
)


def get_brownian_kernel(
    particle_radius: Union[float, NDArray[np.float64]],
    diffusivity_particle: Union[float, NDArray[np.float64]],
    g_collection_term_particle: Union[float, NDArray[np.float64]],
    mean_thermal_speed_particle: Union[float, NDArray[np.float64]],
    alpha_collision_efficiency: Union[float, NDArray[np.float64]] = 1.0,
) -> Union[float, NDArray[np.float64]]:
    """Returns the Brownian coagulation kernel for aerosol particles. Defined
    as the product of the diffusivity of the particles, the collection term
    `g`, and the radius of the particles.

    Args
    ----
    particle_radius : The radius of the particles [m].
    diffusivity_particle : The diffusivity of the particles [m^2/s].
    g_collection_term_particle : The collection term for Brownian coagulation
    [dimensionless].
    alpha_collision_efficiency : The collision efficiency of the particles
    [dimensionless].

    Returns
    -------
    Square matrix of Brownian coagulation kernel for aerosol particles [m^3/s].

    References
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
    Coefficient K12 (with alpha collision efficiency term 13.56)
    """
    # Convert 1D arrays to 2D square matrices
    diffusivity_matrix = np.tile(
        diffusivity_particle, (len(diffusivity_particle), 1)
    )
    radius_matrix = np.tile(particle_radius, (len(particle_radius), 1))
    g_collection_term_matrix = (
        np.tile(
            g_collection_term_particle, (len(g_collection_term_particle), 1)
        )
        ** 2
    )
    mean_thermal_speed_matrix = (
        np.tile(
            mean_thermal_speed_particle, (len(mean_thermal_speed_particle), 1)
        )
        ** 2
    )

    # Sum of diffusivities and radii across particles
    sum_diffusivity = diffusivity_matrix + np.transpose(diffusivity_matrix)
    sum_radius = radius_matrix + np.transpose(radius_matrix)

    # Square root of sums for g-collection terms and mean thermal speeds
    g_term_sqrt = np.sqrt(
        g_collection_term_matrix + np.transpose(g_collection_term_matrix)
    )
    thermal_speed_sqrt = np.sqrt(
        mean_thermal_speed_matrix + np.transpose(mean_thermal_speed_matrix)
    )

    # equation 13.56 from Seinfeld and Pandis
    return (4 * np.pi * sum_diffusivity * sum_radius) / (
        sum_radius / (sum_radius + g_term_sqrt)
        + 4
        * sum_diffusivity
        / (sum_radius * thermal_speed_sqrt * alpha_collision_efficiency)
    )


def get_brownian_kernel_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    mass_particle: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    alpha_collision_efficiency: Union[float, NDArray[np.float64]] = 1.0,
) -> Union[float, NDArray[np.float64]]:
    """Returns the Brownian coagulation kernel for aerosol particles,
    calculating the intermediate properties needed.

    Arguments:
        particle_radius : The radius of the particles [m].
        mass_particle : The mass of the particles [kg].
        temperature : The temperature of the air [K].
        pressure : The pressure of the air [Pa].
        alpha_collision_efficiency : The collision efficiency of the particles
            [dimensionless].

    Returns:
        Square matrix of Brownian coagulation kernel for aerosol particles
            [m^3/s].

    References:
        Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
        physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
        Coefficient K12.
    """
    # calculations to get particle diffusivity
    dynamic_viscosity = get_dynamic_viscosity(temperature)
    air_mean_free_path = get_molecule_mean_free_path(
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=dynamic_viscosity,
    )
    knudsen_number = properties.get_knudsen_number(
        air_mean_free_path,
        particle_radius,
    )
    slip_correction = properties.get_cunningham_slip_correction(
        knudsen_number=knudsen_number
    )
    aerodyanmic_mobility = properties.get_aerodynamic_mobility(
        particle_radius, slip_correction, dynamic_viscosity
    )
    particle_diffusivity = _brownian_diffusivity(
        temperature, aerodyanmic_mobility
    )

    # get thermal speed
    mean_thermal_speed_particle = properties.get_mean_thermal_speed(
        mass_particle, temperature
    )

    # get g collection term
    mean_free_path_particle = _mean_free_path_l(
        diffusivity_particle=particle_diffusivity,
        mean_thermal_speed_particle=mean_thermal_speed_particle,
    )
    g_collection_term_particle = _g_collection_term(
        mean_free_path_particle, particle_radius
    )
    # get the coagulation kernel
    return get_brownian_kernel(
        particle_radius=particle_radius,
        diffusivity_particle=particle_diffusivity,
        g_collection_term_particle=g_collection_term_particle,
        mean_thermal_speed_particle=mean_thermal_speed_particle,
        alpha_collision_efficiency=alpha_collision_efficiency,
    )


def _mean_free_path_l(
    diffusivity_particle: Union[float, NDArray[np.float64]],
    mean_thermal_speed_particle: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the mean free path of particles for coagulation.

    Calculate the mean free path of particles, defined for Brownian
    coagulation as the ratio of the diffusivity of the particles to their mean
    thermal speed. This parameter is crucial for understanding particle
    dynamics in a fluid.

    Args:
    ----
    - diffusivity_particle : The diffusivity of the particles [m^2/s].
    - mean_thermal_speed_particle : The mean thermal speed of the particles
    [m/s].

    Returns:
    -------
    The mean free path of the particles [m].

    References:
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
    Coefficient K12.
    """
    return 8 * diffusivity_particle / (np.pi * mean_thermal_speed_particle)


def _g_collection_term(
    mean_free_path_particle: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Returns the `g` collection term for Brownian coagulation.

    Defined as the ratio of the mean free path of the particles to the
    radius of the particles.

    Args
    ----
    mean_free_path_particle : The mean free path of the particles [m].
    particle_radius : The radius of the particles [m].

    Returns
    -------
    The collection term for Brownian coagulation [dimensionless].

    References
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
    Coefficient K12

    The np.sqrt(2) term appears to be an error in the text, as the term is
    not used in the second edition of the book. And when it it is used, the
    values are too small, by about 2x.
    """
    return (
        (2 * particle_radius + mean_free_path_particle) ** 3
        - (4 * particle_radius**2 + mean_free_path_particle**2) ** (3 / 2)
    ) / (6 * particle_radius * mean_free_path_particle) - 2 * particle_radius


def _brownian_diffusivity(
    temperature: Union[float, NDArray[np.float64]],
    aerodynamic_mobility: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Returns the diffusivity of the particles due to Brownian motion

    THis is just the scaled aerodynamic mobility of the particles.

    Args
    ----
    - temperature : The temperature of the air [K].
    - aerodynamic_mobility : The aerodynamic mobility of the particles [m^2/s].

    Returns
    -------
    The diffusivity of the particles due to Brownian motion [m^2/s].

    References
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
    Coefficient K12
    """
    return float(BOLTZMANN_CONSTANT) * temperature * aerodynamic_mobility
