"""Module for coulomb-related enhancements

References:
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


def get_coulomb_enhancement_ratio(
    radius: Union[float, NDArray[np.float64]],
    charge: Union[int, NDArray[np.float64]] = 0,
    temperature: float = 298.15,
    ratio_lower_limit: float = -200,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Coulomb potential ratio, ϕ_E, for particle-particle interactions.

    The potential ratio is computed using:

    - ϕ_E = - (qᵢ × qⱼ × e²) / [4π ε₀ (rᵢ + rⱼ) k_B T]
        - ϕ_E is the Coulomb potential ratio (dimensionless).
        - qᵢ, qⱼ are the charges (dimensionless, e.g., the number of electrons).
        - e is the elementary charge in coulombs (C).
        - ε₀ is the electric permittivity of free space (F·m⁻¹).
        - rᵢ, rⱼ are the particle radii (m).
        - k_B is the Boltzmann constant (J·K⁻¹).
        - T is the temperature (K).

    Arguments:
        - radius : Radius of the particles (m).
        - charge : Number of charges on the particles (dimensionless).
        - temperature : System temperature (K).
        - ratio_lower_limit : Lower limit to clip the potential ratio for very
          large negative (repulsive) values.

    Returns:
        - The Coulomb potential ratio (dimensionless).

    Examples:
        ``` py title="Example"
        import numpy as np
        from particula.particles.properties.coulomb_enhancement import get_coulomb_enhancement_ratio

        ratio = get_coulomb_enhancement_ratio(
            radius=np.array([1e-7, 2e-7]),
            charge=np.array([1, 2]),
            temperature=298.15,
            ratio_lower_limit=-200
        )
        print(ratio)
        # Output: array([...])
        ```

    References:
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


def get_coulomb_kinetic_limit(
    coulomb_potential: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the kinetic-limit Coulomb enhancement factor, Γₖᵢₙ.

    The kinetic-limit factor is computed by:

    - Γₖᵢₙ =
        1 + ϕ_E      if ϕ_E ≥ 0
        exp(ϕ_E)     if ϕ_E < 0

    where ϕ_E is the Coulomb potential ratio (dimensionless).

    Arguments:
        - coulomb_potential : The Coulomb potential ratio ϕ_E (dimensionless).

    Returns:
        - The Coulomb enhancement factor in the kinetic limit (dimensionless).

    Examples:
        ``` py title="Example"
        import numpy as np
        from particula.particles.properties.coulomb_enhancement import get_coulomb_kinetic_limit

        potential = np.array([-0.5, 0.0, 0.5])
        gamma_kin = get_coulomb_kinetic_limit(potential)
        print(gamma_kin)
        # Output: array([...])
        ```

    References:
        - Equations (6d) and (6e): Gopalakrishnan, R., & Hogan, C. J. (2012).
          Coulomb-influenced collisions in aerosols and dusty plasmas.
          Physical Review E, 85(2). https://doi.org/10.1103/PhysRevE.85.026410
    """
    return np.where(
        coulomb_potential >= 0,
        1 + coulomb_potential,
        safe_exp(coulomb_potential),
    )


def get_coulomb_continuum_limit(
    coulomb_potential: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the continuum-limit Coulomb enhancement factor, Γ_c.

    The continuum-limit factor is computed by:

    - Γ_c =
        ϕ_E / [1 - exp(-ϕ_E)]    if ϕ_E ≠ 0
        1                        if ϕ_E = 0

    where ϕ_E is the Coulomb potential ratio (dimensionless).

    Arguments:
        - coulomb_potential : The Coulomb potential ratio ϕ_E (dimensionless).

    Returns:
        - The Coulomb enhancement factor in the continuum limit (dimensionless).

    Examples:
        ``` py title="Example"
        import numpy as np
        from particula.particles.properties.coulomb_enhancement import get_coulomb_continuum_limit

        potential = np.array([-0.5, 0.0, 0.5])
        gamma_cont = get_coulomb_continuum_limit(potential)
        print(gamma_cont)
        # Output: array([...])
        ```

    References:
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
