"""
The basic Brownian coagulation kernel for aerosol particles, as described by
Seinfeld and Pandis for Fuchs' theory, Chapter 13 table 13.1.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.knudsen_number import calculate_knudsen_number
from particula.constants import BOLTZMANN_CONSTANT
from particula.util.aerodynamic_mobility import particle_aerodynamic_mobility


def mean_thermal_speed(
    mass: Union[float, NDArray[np.float_]],
    temperature: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]:
    """ Returns the particles mean thermal speed. Due to the the impact
    of air molecules on the particles, the particles will have a mean
    thermal speed.

    Args
    ----
    mass : The per particle mass of the particles [kg].
    temperature : The temperature of the air [K].

    Returns
    -------
    The mean thermal speed of the particles [m/s].

    References
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Section 9.5.3 Mean Free Path of an Aerosol Particle Equation 9.87.
    """
    return np.sqrt(
        (8 * BOLTZMANN_CONSTANT.m * temperature)
        / (np.pi * mass),
        dtype=np.float_
    )


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
    mean_thermal_speed_particle : The mean thermal speed of the particles [m/s].

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
    return BOLTZMANN_CONSTANT.m * temperature * aerodynamic_mobility


def brownian_coagulation_kernel(
    radius_particle: Union[float, NDArray[np.float_]],
    diffusivity_particle: Union[float, NDArray[np.float_]],
    g_collection_term_particle: Union[float, NDArray[np.float_]],
    mean_thermal_speed_particle: Union[float, NDArray[np.float_]],
    alpha_collision_efficiency: Union[float, NDArray[np.float_]]
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
