""" Dimensionless coagulation according for several approximations of the
    transition regime.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.particles.properties import coulomb_enhancement


def hard_sphere(
    diffusive_knudsen: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """Hard sphere approximation for the dimensionless coagulation kernel.

    Args:
    -----
    - diffusive_knudsen: The diffusive Knudsen number (K_nD) [dimensionless].

    Returns:
    --------
    The dimensionless coagulation kernel (H) [dimensionless].

    References:
    -----------
    Equations X in:
    - Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
    particles in the transition regime: The effect of the Coulomb potential.
    Journal of Chemical Physics, 126(12).
    https://doi.org/10.1063/1.2713719
    """
    continuum_limit = 4 * np.pi * diffusive_knudsen**2

    fit_constants = [25.836, 11.211, 3.502, 7.211]

    numerator = (
        continuum_limit
        + (fit_constants[0] * diffusive_knudsen**3)
        + ((8 * np.pi) ** (1 / 2) * fit_constants[1] * diffusive_knudsen**4)
    )
    denominator = (
        1
        + (fit_constants[2] * diffusive_knudsen)
        + (fit_constants[3] * diffusive_knudsen**2)
        + (fit_constants[1] * diffusive_knudsen**3)
    )
    return numerator / denominator


def coulomb_dyachkov2007(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Dyachkov et al. (2007) approximation for the dimensionless coagulation
    kernel. Accounts for the Coulomb potential between particles.

    Args:
    -----
    - diffusive_knudsen: The diffusive Knudsen number (K_nD) [dimensionless].
    - coulomb_potential_ratio: The Coulomb potential ratio (phi_E)
    [dimensionless].

    Returns:
    --------
    The dimensionless coagulation kernel (H) [dimensionless].

    References:
    -----------
    Equations X in:
    - Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
    particles in the transition regime: The effect of the Coulomb potential.
    Journal of Chemical Physics, 126(12).
    https://doi.org/10.1063/1.2713719
    """
    coulomb_potential_ratio = np.maximum(
        coulomb_potential_ratio, 1e-16
    )  # Avoid division by zero

    continuum_limit = 4 * np.pi * diffusive_knudsen**2
    kinetic = coulomb_enhancement.kinetic(coulomb_potential_ratio)
    continuum = coulomb_enhancement.continuum(coulomb_potential_ratio)

    ratio_k_c = kinetic / continuum
    adjustment_factor = 1 + diffusive_knudsen * ratio_k_c

    # collected terms
    exponential_decay = np.exp(-coulomb_potential_ratio / adjustment_factor)
    term1 = (
        np.sqrt(2 * np.pi) * diffusive_knudsen * kinetic * exponential_decay
    )

    term2 = adjustment_factor**2 - (
        2 + diffusive_knudsen * ratio_k_c
    ) * diffusive_knudsen * ratio_k_c * np.exp(
        -coulomb_potential_ratio
        / (adjustment_factor * (2 + diffusive_knudsen * ratio_k_c))
    )
    term3 = 1 - exponential_decay
    term4 = 1 - np.exp(-coulomb_potential_ratio)

    # Using vectorized operations to combine terms
    return continuum_limit / (term1 / term2 + term3 / term4)


def coulomb_gatti2008(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Gatti et al. (2008) approximation for the dimensionless coagulation
    kernel. Accounts for the Coulomb potential between particles.

    Args:
    -----
    - diffusive_knudsen: The diffusive Knudsen number (K_nD) [dimensionless].
    - coulomb_potential_ratio: The Coulomb potential ratio (phi_E)
    [dimensionless].

    Returns:
    --------
    The dimensionless coagulation kernel (H) [dimensionless].

    References:
    -----------
    - Equations X in:
    Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
    charging in plasmas over a wide range of collisionality. Physical Review
    E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
    https://doi.org/10.1103/PhysRevE.78.046402
    """
    # Ensure no division by zero
    coulomb_potential_ratio = np.maximum(coulomb_potential_ratio, 1e-16)

    kinetic = coulomb_enhancement.kinetic(coulomb_potential_ratio)
    continuum = coulomb_enhancement.continuum(coulomb_potential_ratio)

    continuum_limit = 4 * np.pi * diffusive_knudsen**2
    pi_sqrt = np.sqrt(np.pi)

    factored_term = (pi_sqrt * continuum * coulomb_potential_ratio * 1.22) / (
        2 * kinetic * diffusive_knudsen
    )
    exponential_decay = np.exp(-factored_term)

    term1 = continuum_limit * (1 - (1 + factored_term) * exponential_decay)
    term2 = (
        np.sqrt(8 * np.pi)
        * diffusive_knudsen
        * (
            1
            + (
                2
                * pi_sqrt
                * (1.22**3)
                * continuum
                * (coulomb_potential_ratio**3)
            )
            / (9 * (kinetic**2) * diffusive_knudsen)
        )
        * exponential_decay
    )
    # coulomb_potential_ratio is below 0 returns hard sphere
    return np.where(
        coulomb_potential_ratio > 0,
        term1 + term2,
        hard_sphere(diffusive_knudsen),
    )


def coulomb_gopalakrishnan2012(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Gopalakrishnan and Hogan (2012) approximation for the dimensionless
    coagulation kernel. Accounts for the Coulomb potential between particles.

    Args:
    -----
    - diffusive_knudsen: The diffusive Knudsen number (K_nD) [dimensionless].
    - coulomb_potential_ratio: The Coulomb potential ratio (phi_E)
    [dimensionless].

    Returns:
    --------
    The dimensionless coagulation kernel (H) [dimensionless].

    References:
    -----------
    - Equations X in:
    Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
    in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear,
    and Soft Matter Physics, 85(2).
    https://doi.org/10.1103/PhysRevE.85.026410
    """
    continuum_limit = 4 * np.pi * diffusive_knudsen**2
    min_fxn = np.minimum(
        diffusive_knudsen,
        3 * diffusive_knudsen / (2 * coulomb_potential_ratio),
    )
    # Condition for the transition regime
    condition = (coulomb_potential_ratio > 0.5) & (min_fxn < 2.5)
    return np.where(
        condition,
        continuum_limit / (1 + 1.598 * min_fxn**1.1709),
        hard_sphere(diffusive_knudsen),
    )


def coulomb_chahl2019(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Chahl and Gopalakrishnan (2019) approximation for the dimensionless
    coagulation kernel. Accounts for the Coulomb potential between particles.

    Args:
    -----
    - diffusive_knudsen: The diffusive Knudsen number (K_nD) [dimensionless].
    - coulomb_potential_ratio: The Coulomb potential ratio (phi_E)
    [dimensionless].

    Returns:
    --------
    The dimensionless coagulation kernel (H) [dimensionless].

    References:
    -----------
    - Equations X in:
    Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
    molecular regime Coulombic collisions in aerosols and dusty plasmas.
    Aerosol Science and Technology, 53(8), 933-957.
    https://doi.org/10.1080/02786826.2019.1614522
    """
    # Ensure no division by zero
    coulomb_potential_ratio = np.maximum(coulomb_potential_ratio, 1e-12)

    correction0 = 2.5
    correction1 = 4.528 * np.exp(
        -1.088 * coulomb_potential_ratio
    ) + 0.7091 * np.log(1 + 1.527 * coulomb_potential_ratio)
    correction2 = 11.36 * (coulomb_potential_ratio**0.272) - 10.33
    correction3 = -0.003533 * coulomb_potential_ratio + 0.05971
    diff_knudsen_log = np.log(diffusive_knudsen)

    correction_mu = (correction2 / correction0) * (
        (1 + correction3 * (diff_knudsen_log - correction1) / correction0)
        ** (-1 / correction3 - 1)
        * np.exp(
            -1
            * (
                1
                + correction3 * (diff_knudsen_log - correction1) / correction0
            )
            ** (-1 / correction3)
        )
    )
    return np.where(
        coulomb_potential_ratio > 0,
        np.exp(correction_mu) * hard_sphere(diffusive_knudsen),
        hard_sphere(diffusive_knudsen),
    )
