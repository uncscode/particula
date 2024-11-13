"""Radius of collision module for non-spherical particles."""

from typing import Union
import numpy as np
from numpy.typing import NDArray


def mulholland_1988(
    radius_giration: Union[NDArray[np.float64], float]
) -> Union[NDArray[np.float64], float]:
    """Collision radius is equal to the radius of gyration.

    Args:
        radius_giration: Radius of gyration of the particle [m].

    Returns:
        (float or NDArray[float]): Collision radius of the particle [m].

    References:
        Mulholland, G. W., Mountain, R. D., Samson, R. J., & Ernst, M. H.
        (1988). Cluster Size Distribution for Free Molecular Agglomeration.
        Energy and Fuels, 2(4). https://doi.org/10.1021/ef00010a014
    """
    return radius_giration


def rogak_flagan_1992(
    radius_giration: Union[NDArray[np.float64], float],
    fractal_dimension: Union[NDArray[np.float64], float],
) -> Union[NDArray[np.float64], float]:
    """Collision radius with fractal dimension by Rogak and Flagan 1992.

    Args:
        radius_giration: Radius of gyration of the particle [m].
        fractal_dimension: Fractal dimension of the particle
            [dimensionless, df].

    Returns:
        (float or NDArray[float]): Collision radius of the particle [m].

    References:
        Rogak, S. N., & Flagan, R. C. (1992). Coagulation of aerosol
        agglomerates in the transition regime. Journal of Colloid and
        Interface Science, 151(1), 203-224.
        https://doi.org/10.1016/0021-9797(92)90252-H
    """
    return np.sqrt((fractal_dimension + 2) / 3) * radius_giration


def zurita_gotor_2002(
    radius_giration: Union[NDArray[np.float64], float],
    fractal_prefactor: Union[NDArray[np.float64], float],
) -> Union[NDArray[np.float64], float]:
    """Collision radius according to Zurita-Gotor and Rosner (2002).

    Args:
        radius_giration: Radius of gyration of the particle [m].
        fractal_prefactor: Fractal prefactor of the particle
            [dimensionless, k0].

    Returns:
        (float or NDArray[float]): Collision radius of the particle [m].

    References:
        Zurita-Gotor, M., & Rosner, D. E. (2002). Effective diameters for
        collisions of fractal-like aggregates: Recommendations for improved
        aerosol coagulation frequency predictions. Journal of Colloid and
        Interface Science, 255(1).
        https://doi.org/10.1006/jcis.2002.8634
    """
    return 1.037 * (fractal_prefactor**0.077) * radius_giration


def thajudeen_rc(sigma: float, df: float, n: float, rg: float) -> float:
    """Collision radius according to Thajudeen et al. (2012)."""
    alpha1 = 0.253 * df**2 - 1.209 * df + 1.433
    alpha2 = -0.218 * df**2 + 0.964 * df - 0.180
    phi = 1 / (alpha1 * np.log(n) + alpha2)
    rs_i = phi * rg
    rs_ii = (sigma * (1.203 - 0.4315 / df) / 2) * (4 * rs_i / sigma) ** (
        0.8806 + 0.3497 / df
    )
    return rs_ii / 2


def model_rg(c1: float, c6: float, rg_tilde: float) -> float:
    """New model Rc with Rg parameter."""
    return c1 * rg_tilde + c6


def model_rg_df(
    c1: float, c2: float, c6: float, df: float, rg_tilde: float
) -> float:
    """New model Rc with Rg and Df parameters."""
    return c1 * (df**c2) * rg_tilde + c6


def model_rg_df_k0(
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    c6: float,
    df: float,
    k0: float,
    rg_tilde: float,
) -> float:
    """New model Rc with Rg, Df, and k0 parameters."""
    return c1 * (df**c2) * (k0**c3) * rg_tilde + c4 * k0 + c6


def model_rg_df_k0_a13(
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    c5: float,
    c6: float,
    df: float,
    k0: float,
    a13: float,
    rg_tilde: float,
) -> float:
    """New model Rc with Rg, Df, k0, and A13 parameters."""
    return c1 * (df**c2) * (k0**c3) * rg_tilde + c4 * k0 + c5 * a13 + c6
