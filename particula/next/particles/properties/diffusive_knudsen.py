""" Calculate the diffusive knudsen number
"""
from typing import Union
from numpy.typing import NDArray
import numpy as np
from particula.constants import BOLTZMANN_CONSTANT

from particula.next.particles.properties import coulomb_enhancement
from particula.util.reduced_quantity import reduced_value


def diffusive_knudsen_number(
    radius: Union[float, NDArray[np.float_]],
    mass_particle: Union[float, NDArray[np.float_]],
    friction_factor: Union[float, NDArray[np.float_]],
    coulomb_potential_ratio: Union[float, NDArray[np.float_]] = 0.0,
    temperature: float = 298.15
) -> Union[float, NDArray[np.float_]]:
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
    if isinstance(radius, np.ndarray):
        # square matrix of radius
        radius = np.array(radius)
        radius = np.tile(radius, (len(radius), 1))
    if isinstance(mass_particle, np.ndarray):
        # square matrix of mass
        mass_particle = np.array(mass_particle)
        mass_particle = np.tile(mass_particle, (len(mass_particle), 1))
    if isinstance(friction_factor, np.ndarray):
        # square matrix of charge
        friction_factor = np.array(friction_factor)
        friction_factor = np.tile(friction_factor, (len(friction_factor), 1))

    # reduced values
    reduced_mass = reduced_value(
        mass_particle, np.transpose(mass_particle))
    reduced_friction_factor = reduced_value(
        friction_factor, np.transpose(friction_factor))
    # radius sum
    sum_radius = radius + np.transpose(radius)
    return (
        ((temperature * BOLTZMANN_CONSTANT.m * reduced_mass)**0.5
         / reduced_friction_factor)
        /
        (sum_radius * coulomb_enhancement.continuum(coulomb_potential_ratio)
         / coulomb_enhancement.kinetic(coulomb_potential_ratio))
    )
