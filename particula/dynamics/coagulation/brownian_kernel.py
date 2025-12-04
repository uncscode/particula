"""The basic Brownian coagulation kernel for aerosol particles.

This module provides functions to calculate the Brownian coagulation
kernel for aerosol particles, based on Fuchs' theory as described by
Seinfeld and Pandis (2016), Chapter 13, Table 13.1.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula import gas, particles
from particula.util.constants import BOLTZMANN_CONSTANT


def get_brownian_kernel(
    particle_radius: Union[float, NDArray[np.float64]],
    diffusivity_particle: Union[float, NDArray[np.float64]],
    g_collection_term_particle: Union[float, NDArray[np.float64]],
    mean_thermal_speed_particle: Union[float, NDArray[np.float64]],
    alpha_collision_efficiency: Union[float, NDArray[np.float64]] = 1.0,
) -> Union[float, NDArray[np.float64]]:
    """Calculate the Brownian coagulation kernel for aerosol particles.

    This function computes the Brownian coagulation kernel, which is
    defined as the product of the diffusivity of the particles, the
    collection term `g`, and the radius of the particles. The equation
    used is:

    - K = (4π × D × r) / (r / (r + g) + 4D / (r × v × α))
        - K is the Brownian coagulation kernel [m³/s].
        - D is the diffusivity of the particles [m²/s].
        - r is the radius of the particles [m].
        - g is the collection term for Brownian coagulation [dimensionless].
        - v is the mean thermal speed of the particles [m/s].
        - α is the collision efficiency of the particles [dimensionless].

    Arguments:
        - particle_radius : The radius of the particles [m].
        - diffusivity_particle : The diffusivity of the particles [m²/s].
        - g_collection_term_particle : The collection term for Brownian
          coagulation [dimensionless].
        - mean_thermal_speed_particle : The mean thermal speed of the
          particles [m/s].
        - alpha_collision_efficiency : The collision efficiency of the
          particles [dimensionless].

    Returns:
        - Square matrix of Brownian coagulation kernel for aerosol particles
          [m³/s].

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
          physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
          Coefficient K12 (with alpha collision efficiency term 13.56).
    """
    # Convert 1D arrays to 2D square matrices
    # Type narrowing: convert scalar to array if needed for len() compatibility
    diffusivity_arr: NDArray[np.float64] = np.atleast_1d(diffusivity_particle)
    radius_arr: NDArray[np.float64] = np.atleast_1d(particle_radius)
    g_collection_arr: NDArray[np.float64] = np.atleast_1d(
        g_collection_term_particle
    )
    mean_thermal_speed_arr: NDArray[np.float64] = np.atleast_1d(
        mean_thermal_speed_particle
    )

    diffusivity_matrix = np.tile(diffusivity_arr, (len(diffusivity_arr), 1))
    radius_matrix = np.tile(radius_arr, (len(radius_arr), 1))
    g_collection_term_matrix = (
        np.tile(g_collection_arr, (len(g_collection_arr), 1)) ** 2
    )
    mean_thermal_speed_matrix = (
        np.tile(mean_thermal_speed_arr, (len(mean_thermal_speed_arr), 1)) ** 2
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
    particle_mass: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    alpha_collision_efficiency: Union[float, NDArray[np.float64]] = 1.0,
) -> Union[float, NDArray[np.float64]]:
    """Calculate the Brownian coagulation kernel using system state parameters.

    This function calculates the Brownian coagulation kernel for aerosol
    particles by determining the necessary intermediate properties such as
    particle diffusivity and mean thermal speed. The equation used is:

    - K = (4π × D × r) / (r / (r + g) + 4D / (r × v × α))
        - K is the Brownian coagulation kernel [m³/s].
        - D is the diffusivity of the particles [m²/s].
        - r is the radius of the particles [m].
        - g is the collection term for Brownian coagulation [dimensionless].
        - v is the mean thermal speed of the particles [m/s].
        - α is the collision efficiency of the particles [dimensionless].

    Arguments:
        - particle_radius : The radius of the particles [m].
        - particle_mass : The mass of the particles [kg].
        - temperature : The temperature of the air [K].
        - pressure : The pressure of the air [Pa].
        - alpha_collision_efficiency : The collision efficiency of the
          particles [dimensionless].

    Returns:
        - Square matrix of Brownian coagulation kernel for aerosol particles
          [m³/s].

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
          physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
          Coefficient K12.
    """
    # calculations to get particle diffusivity
    dynamic_viscosity = gas.get_dynamic_viscosity(temperature)
    air_mean_free_path = gas.get_molecule_mean_free_path(
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=dynamic_viscosity,
    )
    knudsen_number = particles.get_knudsen_number(
        air_mean_free_path,
        particle_radius,
    )
    slip_correction = particles.get_cunningham_slip_correction(
        knudsen_number=knudsen_number
    )
    aerodyanmic_mobility = particles.get_aerodynamic_mobility(
        particle_radius, slip_correction, dynamic_viscosity
    )
    particle_diffusivity = _brownian_diffusivity(
        temperature, aerodyanmic_mobility
    )

    # get thermal speed
    mean_thermal_speed_particle = particles.get_mean_thermal_speed(
        particle_mass, temperature
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
    """Calculate the mean free path of particles for coagulation.

    Calculate the mean free path of particles for coagulation.

    This function calculates the mean free path of particles, defined for
    Brownian coagulation as the ratio of the diffusivity of the particles
    to their mean thermal speed. This parameter is crucial for understanding
    particle dynamics in a fluid. The equation used is:

    - λ = (8 × D) / (π × v)
        - λ is the mean free path of the particles [m].
        - D is the diffusivity of the particles [m²/s].
        - v is the mean thermal speed of the particles [m/s].

    Arguments:
        - diffusivity_particle : The diffusivity of the particles [m²/s].
        - mean_thermal_speed_particle : The mean thermal speed of the
          particles [m/s].

    Returns:
        - The mean free path of the particles [m].

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
          physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
          Coefficient K12.
    """
    return 8 * diffusivity_particle / (np.pi * mean_thermal_speed_particle)


def _g_collection_term(
    mean_free_path_particle: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the `g` collection term for Brownian coagulation.

    This function calculates the `g` collection term for Brownian
    coagulation, defined as the ratio of the mean free path of the particles
    to the radius of the particles. The equation used is:

    - g = ((2r + λ)³ - (4r² + λ²)^(3/2)) / (6rλ) - 2r
        - g is the collection term for Brownian coagulation [dimensionless].
        - λ is the mean free path of the particles [m].
        - r is the radius of the particles [m].

    Arguments:
        - mean_free_path_particle : The mean free path of the particles [m].
        - particle_radius : The radius of the particles [m].

    Returns:
        - The collection term for Brownian coagulation [dimensionless].

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
          physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
          Coefficient K12.

    Note:
        The np.sqrt(2) term appears to be an error in the text, as the term is
        not used in the second edition of the book. When it is used, the values
        are too small, by about 2x.
    """
    return (
        (2 * particle_radius + mean_free_path_particle) ** 3
        - (4 * particle_radius**2 + mean_free_path_particle**2) ** (3 / 2)
    ) / (6 * particle_radius * mean_free_path_particle) - 2 * particle_radius


def _brownian_diffusivity(
    temperature: Union[float, NDArray[np.float64]],
    aerodynamic_mobility: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the diffusivity of particles due to Brownian motion.

    This function calculates the diffusivity of particles due to Brownian
    motion, which is essentially the scaled aerodynamic mobility of the
    particles. The equation used is:

    - D = k × T × B
        - D is the diffusivity of the particles [m²/s].
        - k is the Boltzmann constant [J/K].
        - T is the temperature of the air [K].
        - B is the aerodynamic mobility of the particles [m²/s].

    Arguments:
        - temperature : The temperature of the air [K].
        - aerodynamic_mobility : The aerodynamic mobility of the particles
          [m²/s].

    Returns:
        - The diffusivity of the particles due to Brownian motion [m²/s].

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
          physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
          Coefficient K12.
    """
    return float(BOLTZMANN_CONSTANT) * temperature * aerodynamic_mobility
