"""
The basic Brownian coagulation kernel for aerosol particles, as described by
Seinfeld and Pandis for Fuchs' theory, Chapter 13 table 13.1.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.constants import BOLTZMANN_CONSTANT
from particula.next.particles import properties
from particula.util.mean_free_path import molecule_mean_free_path
from particula.util.dynamic_viscosity import dyn_vis  # pyright: ignore


def mean_free_path_l(
    diffusivity_particle: Union[float, NDArray[np.float_]],
    mean_thermal_speed_particle: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]:
    """ Returns the mean free path of the particles. Defined for Brownian
    coagulation as the ratio of the diffusivity of the particles to the
    mean thermal speed of the particles.

    Args
    ----
    diffusivity_particle : The diffusivity of the particles [m^2/s].
    mean_thermal_speed_particle : The mean thermal speed of the particles [m/s]

    Returns
    -------
    The mean free path of the particles [m].

    References
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
    Coefficient K12
    """
    return 8 * diffusivity_particle / (np.pi * mean_thermal_speed_particle)


def g_collection_term(
    mean_free_path_particle: Union[float, NDArray[np.float_]],
    radius_particle: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]:
    """ Returns the `g` collection term for Brownian coagulation. Defined as
    the ratio of the mean free path of the particles to the radius of the
    particles.

    Args
    ----
    mean_free_path_particle : The mean free path of the particles [m].
    radius_particle : The radius of the particles [m].

    Returns
    -------
    The collection term for Brownian coagulation [dimensionless].

    References
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
    Coefficient K12
    """
    return (
        (2 * radius_particle - mean_free_path_particle)**3
        - (4 * radius_particle**2 + mean_free_path_particle**2) ** (3/2)
        - 2 * radius_particle
    ) / (6 * radius_particle - mean_free_path_particle)


def brownian_diffusivity(
    temperature: Union[float, NDArray[np.float_]],
    aerodynamic_mobility: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]:
    """ Returns the diffusivity of the particles due to Brownian motion. Which
    is just the scaled aerodynamic mobility of the particles.

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
    return float(BOLTZMANN_CONSTANT.m) * temperature * aerodynamic_mobility


def brownian_coagulation_kernel(
    radius_particle: Union[float, NDArray[np.float_]],
    diffusivity_particle: Union[float, NDArray[np.float_]],
    g_collection_term_particle: Union[float, NDArray[np.float_]],
    mean_thermal_speed_particle: Union[float, NDArray[np.float_]],
    alpha_collision_efficiency: Union[float, NDArray[np.float_]] = 1.0
) -> Union[float, NDArray[np.float_]]:
    """ Returns the Brownian coagulation kernel for aerosol particles. Defined
    as the product of the diffusivity of the particles, the collection term
    `g`, and the radius of the particles.

    Args
    ----
    radius_particle : The radius of the particles [m].
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
    # pre-compute some terms
    sum_diffusivity = diffusivity_particle + np.transpose(diffusivity_particle)
    sum_radius = radius_particle + np.transpose(radius_particle)
    g_term_sqrt = np.sqrt(
        g_collection_term_particle**2
        + np.transpose(g_collection_term_particle**2)
    )
    thermal_speed_sqrt = np.sqrt(
        mean_thermal_speed_particle**2
        + np.transpose(mean_thermal_speed_particle**2)
    )
    # equation 13.56 from Seinfeld and Pandis
    return (
        4 * np.pi * sum_diffusivity * sum_radius
    ) / (
        sum_radius / (sum_radius + g_term_sqrt)
        + 4 * sum_diffusivity
        / (sum_radius * thermal_speed_sqrt * alpha_collision_efficiency)
        )


def system_state_brownian_coagulation_kernel(
    radius_particle: Union[float, NDArray[np.float_]],
    mass_particle: Union[float, NDArray[np.float_]],
    temperature: float,
    pressure: float,
    alpha_collision_efficiency: Union[float, NDArray[np.float_]] = 1.0
) -> Union[float, NDArray[np.float_]]:
    """ Returns the Brownian coagulation kernel for aerosol particles.

    Args
    ----
    radius_particle : The radius of the particles [m].
    mass_particle : The mass of the particles [kg].
    temperature : The temperature of the air [K].
    pressure : The pressure of the air [Pa].
    alpha_collision_efficiency : The collision efficiency of the particles
    [dimensionless].

    Returns
    -------
    Square matrix of Brownian coagulation kernel for aerosol particles [m^3/s].

    References
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
    Coefficient K12
    """
    # calculations to get particle diffusivity
    dynamic_viscosity = float(dyn_vis(temperature).m)  # pyright: ignore
    air_mean_free_path = molecule_mean_free_path(
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=dynamic_viscosity
    )
    knudsen_number = properties.calculate_knudsen_number(
        air_mean_free_path,
        radius_particle,
    )
    slip_correction = properties.cunningham_slip_correction(
        knudsen_number=knudsen_number)
    aerodyanmic_mobility = properties.particle_aerodynamic_mobility(
        radius_particle, slip_correction, dynamic_viscosity)
    particle_diffusivity = brownian_diffusivity(
        temperature, aerodyanmic_mobility)

    # get thermal speed
    mean_thermal_speed_particle = properties.mean_thermal_speed(
        mass_particle,
        temperature)

    # get g collection term
    mean_free_path_particle = mean_free_path_l(
        diffusivity_particle=particle_diffusivity,
        mean_thermal_speed_particle=mean_thermal_speed_particle
    )
    g_collection_term_particle = g_collection_term(
        mean_free_path_particle, radius_particle
    )
    # get the coagulation kernel
    return brownian_coagulation_kernel(
        radius_particle=radius_particle,
        diffusivity_particle=particle_diffusivity,
        g_collection_term_particle=g_collection_term_particle,
        mean_thermal_speed_particle=mean_thermal_speed_particle,
        alpha_collision_efficiency=alpha_collision_efficiency
    )
