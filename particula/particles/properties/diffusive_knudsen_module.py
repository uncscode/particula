""" Module for the diffusive knudsen number
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np
from particula.constants import BOLTZMANN_CONSTANT

from particula.particles.properties import coulomb_enhancement
from particula.util.reduced_quantity import reduced_self_broadcast


def diffusive_knudsen_number(
    radius: Union[float, NDArray[np.float64]],
    mass_particle: Union[float, NDArray[np.float64]],
    friction_factor: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]] = 0.0,
    temperature: float = 298.15,
) -> Union[float, NDArray[np.float64]]:
    """
    Diffusive Knudsen number. The *diffusive* Knudsen number is different
    from Knudsen number. Ratio of: mean persistence of one particle to the
    effective length scale of particle--particle Coulombic interaction

    Args:
    -----
    - radius: The radius of the particle [m].
    - mass_particle: The mass of the particle [kg].
    - friction_factor: The friction factor of the particle [dimensionless].
    - coulomb_potential_ratio: The Coulomb potential ratio, zero if
     no charges [dimensionless].
    - temperature: The temperature of the system [K].

    Returns:
    --------
    The diffusive Knudsen number [dimensionless], as a square matrix, of all
    particle-particle interactions.

    References:
    -----------
    - Equation 5 in, with charges:
    Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
    molecular regime Coulombic collisions in aerosols and dusty plasmas.
    Aerosol Science and Technology, 53(8), 933-957.
    https://doi.org/10.1080/02786826.2019.1614522
    - Equation 3b in, no charges:
    Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
    in aerosols and dusty plasmas. Physical Review E - Statistical,
    Nonlinear, and Soft Matter Physics, 85(2).
    https://doi.org/10.1103/PhysRevE.85.026410
    """
    # Calculate the pairwise sum of radii
    if isinstance(radius, np.ndarray):
        sum_of_radii = radius[:, np.newaxis] + radius
    else:
        sum_of_radii = 2 * radius

    # Calculate reduced mass
    if isinstance(mass_particle, np.ndarray):
        reduced_mass = reduced_self_broadcast(mass_particle)
    else:
        reduced_mass = mass_particle

    # Calculate reduced friction factor
    if isinstance(friction_factor, np.ndarray):
        reduced_friction_factor = reduced_self_broadcast(friction_factor)
    else:
        reduced_friction_factor = friction_factor

    # Calculate the kinetic and continuum enhancements
    kinetic_enhance = coulomb_enhancement.kinetic(coulomb_potential_ratio)
    continuum_enhance = coulomb_enhancement.continuum(coulomb_potential_ratio)

    # Final calculation of diffusive Knudsen number
    numerator = (
        np.sqrt(temperature * BOLTZMANN_CONSTANT.m * reduced_mass)
        / reduced_friction_factor
    )
    denominator = sum_of_radii * continuum_enhance / kinetic_enhance

    return numerator / denominator
