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

from particula.util.constants import (
    BOLTZMANN_CONSTANT,
    ELECTRIC_PERMITTIVITY,
    ELEMENTARY_CHARGE_VALUE,
)
from particula.util.machine_limit import safe_exp


def ratio(
    radius: Union[float, NDArray[np.float64]],
    charge: Union[int, NDArray[np.float64]] = 0,
    temperature: float = 298.15,
    ratio_lower_limit: float = -200,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Coulomb potential ratio, ϕ_E, for particle-particle
    interactions.

    The function is given by:

        ϕ_E = - (q_i q_j e²) / [4 π ε₀ (r_i + r_j) k_B T]

    where:

    - q_i, q_j : Charges on particles i and j [dimensionless].
    - e        : Elementary charge [C].
    - ε₀       : Electric permittivity of free space [F·m⁻¹].
    - r_i, r_j : Radii of particles i and j [m].
    - k_B      : Boltzmann constant [J·K⁻¹].
    - T        : Temperature [K].

    Arguments:
    ----------
        - radius : Radius of the particles [m].
        - charge : Number of charges on the particles [dimensionless].
        - temperature : Temperature of the system [K].
        - ratio_lower_limit : Lower limit for the Coulomb potential ratio.
            This is used to clip the ratio to avoid numerical issues, in
            subsequent kernel calculations. This only applies to highly
            negative values of the ratio, which are high repulsion cases.

    Returns:
    --------
        - coulomb_potential_ratio : The Coulomb potential ratio ϕ_E
            [dimensionless].

    References:
    -----------
        - Equation (7): Gopalakrishnan, R., & Hogan, C. J. (2012).
          Coulomb-influenced collisions in aerosols and dusty plasmas.
          Physical Review E, 85(2). https://doi.org/10.1103/PhysRevE.85.026410
    """
    if isinstance(radius, np.ndarray):
        # square matrix of radius
        radius = np.array(radius)
        radius = np.tile(radius, (len(radius), 1))
        # square matrix of charge
        charge = np.array(charge)
        charge = np.tile(charge, (len(charge), 1))

    numerator = (
        -1 * charge * np.transpose(charge) * (ELEMENTARY_CHARGE_VALUE**2)
    )
    denominator = (
        4 * np.pi * ELECTRIC_PERMITTIVITY * (radius + np.transpose(radius))
    )
    coulomb_potential_ratio = numerator / (
        denominator * BOLTZMANN_CONSTANT * temperature
    )
    return np.clip(
        coulomb_potential_ratio, ratio_lower_limit, np.finfo(np.float64).max
    )


def kinetic(
    coulomb_potential: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the kinetic limit of the Coulombic enhancement factor.

    The function is given by:

        Γ_kinetic =
            1 + ϕ_E          if ϕ_E ≥ 0
            exp(ϕ_E)         if ϕ_E < 0

    Arguments:
    ----------
        - coulomb_potential : The Coulomb potential ratio ϕ_E [dimensionless].

    Returns:
    --------
        - gamma_kinetic : Coulomb enhancement factor in the kinetic limit
            [dimensionless].

    References:
    -----------
        - Equations (6d) and (6e): Gopalakrishnan, R., & Hogan, C. J. (2012).
          Coulomb-influenced collisions in aerosols and dusty plasmas.
          Physical Review E, 85(2). https://doi.org/10.1103/PhysRevE.85.026410
    """
    return np.where(
        coulomb_potential >= 0,
        1 + coulomb_potential,
        safe_exp(coulomb_potential),
    )


def continuum(
    coulomb_potential: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the continuum limit of the Coulombic enhancement factor.

    The function is given by:

        Γ_continuum =
            ϕ_E / [1 - exp(-ϕ_E)]    if ϕ_E ≠ 0
            1                         if ϕ_E = 0

    Arguments:
    ----------
        - coulomb_potential : The Coulomb potential ratio ϕ_E [dimensionless].

    Returns:
    --------
        - gamma_continuum : Coulomb enhancement factor in the continuum limit
            [dimensionless].

    References:
    -----------
        - Equation (6b): Gopalakrishnan, R., & Hogan, C. J. (2012).
          Coulomb-influenced collisions in aerosols and dusty plasmas.
          Physical Review E, 85(2). https://doi.org/10.1103/PhysRevE.85.026410
    """
    denominator = 1 - safe_exp(-1 * coulomb_potential)
    return np.divide(
        coulomb_potential,
        denominator,
        out=np.ones_like(denominator),
        where=denominator != 0,
    )
