"""Module for coulomb-related enhancements

References:
----------
Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear,
and Soft Matter Physics, 85(2).
https://doi.org/10.1103/PhysRevE.85.026410
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.constants import (
    BOLTZMANN_CONSTANT,
    ELECTRIC_PERMITTIVITY,
    ELEMENTARY_CHARGE_VALUE,
)


def ratio(
    radius: Union[float, NDArray[np.float64]],
    charge: Union[int, NDArray[np.float64]] = 0,
    temperature: float = 298.15,
) -> Union[float, NDArray[np.float64]]:
    """Calculates the Coulomb potential ratio, phi_E. For all particle-
    particle interactions.

    Args:
        radius: The radius of the particle [m].
        charge: The number of charges on the particle [dimensionless].
        temperature: The temperature of the system [K].

    Returns:
        The Coulomb potential ratio [dimensionless].

    References:
        Equation (7): Gopalakrishnan, R., & Hogan, C. J. (2012).
            Coulomb-influenced collisions in aerosols and dusty plasmas.
            Physical Review E - Statistical, Nonlinear, and Soft Matter
            Physics, 85(2). (https://doi.org/10.1103/PhysRevE.85.026410)
    """
    if isinstance(radius, np.ndarray):
        # square matrix of radius
        radius = np.array(radius)
        radius = np.tile(radius, (len(radius), 1))
        # square matrix of charge
        charge = np.array(charge)
        charge = np.tile(charge, (len(charge), 1))

    numerator = (
        -1 * charge * np.transpose(charge) * (ELEMENTARY_CHARGE_VALUE.m**2)
    )
    denominator = (
        4 * np.pi * ELECTRIC_PERMITTIVITY.m * (radius + np.transpose(radius))
    )
    return numerator / (denominator * BOLTZMANN_CONSTANT.m * temperature)


def kinetic(
    coulomb_potential: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """Calculates the Coulombic enhancement kinetic limit. For all particle-
    particle interactions.

    Args:
        coulomb_potential: The Coulomb potential ratio [dimensionless].

    Returns:
        The Coulomb enhancement for the kinetic limit [dimensionless].

    References:
        Equation 6d and 6e in, Gopalakrishnan, R., & Hogan, C. J. (2012).
        Coulomb-influenced collisions in aerosols and dusty plasmas.
        Physical Review E - Statistical, Nonlinear,
        and Soft Matter Physics, 85(2).
        (https://doi.org/10.1103/PhysRevE.85.026410)
    """
    # return 1 + coulumb_potential if ratio >=0,
    # otherwise np.exp(coulomb_potential)
    return np.where(
        coulomb_potential >= 0,
        1 + coulomb_potential,
        np.exp(coulomb_potential),
    )


def continuum(
    coulomb_potential: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """Calculates the Coulombic enhancement continuum limit. For all particle-
    particle interactions.

    Args:
        coulomb_potential: The Coulomb potential ratio [dimensionless].

    Returns:
        The Coulomb enhancement for the continuum limit [dimensionless].

    References:
        Equation 6b in: Gopalakrishnan, R., & Hogan, C. J. (2012).
        Coulomb-influenced collisions in aerosols and dusty plasmas.
        Physical Review E - Statistical, Nonlinear,
        and Soft Matter Physics, 85(2).
        (https://doi.org/10.1103/PhysRevE.85.026410)
    """
    # return coulomb_potential/(1-np.exp(-1*coulomb_potential)) if ratio != 0,
    # otherwise 1
    denominator = 1 - np.exp(-1 * coulomb_potential)
    return np.divide(
        coulomb_potential,
        denominator,
        out=np.ones_like(denominator),
        where=denominator != 0,
    )
