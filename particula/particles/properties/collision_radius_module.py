"""Radius of collision module for non-spherical particles. Based on the
compiled models in Qian et al. (2022).

Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
Effects of agglomerate characteristics on their collision kernels in
the free molecular regime. Journal of Aerosol Science, 159.
https://doi.org/10.1016/j.jaerosci.2021.10586
"""

from typing import Union, Tuple
import numpy as np
from numpy.typing import NDArray


def mulholland_1988(
    radius_gyration: Union[NDArray[np.float64], float]
) -> Union[NDArray[np.float64], float]:
    """Collision radius is equal to the radius of gyration.

    Args:
        radius_gyration: Radius of gyration of the particle [m].

    Returns:
        (float or NDArray[float]): Collision radius of the particle [m].

    References:
        Mulholland, G. W., Mountain, R. D., Samson, R. J., & Ernst, M. H.
        (1988). Cluster Size Distribution for Free Molecular Agglomeration.
        Energy and Fuels, 2(4). https://doi.org/10.1021/ef00010a014

    Examples:
        ``` py title="Example"
        mulholland_1988(1.5)
        # 1.5
        ```
    """
    return radius_gyration


def rogak_flagan_1992(
    radius_gyration: Union[NDArray[np.float64], float],
    fractal_dimension: Union[NDArray[np.float64], float],
) -> Union[NDArray[np.float64], float]:
    """Collision radius with fractal dimension by Rogak and Flagan 1992.

    Args:
        radius_gyration: Radius of gyration of the particle [m].
        fractal_dimension: Fractal dimension of the particle
            [dimensionless, df].

    Returns:
        (float or NDArray[float]): Collision radius of the particle [m].

    References:
        Rogak, S. N., & Flagan, R. C. (1992). Coagulation of aerosol
        agglomerates in the transition regime. Journal of Colloid and
        Interface Science, 151(1), 203-224.
        https://doi.org/10.1016/0021-9797(92)90252-H

    Examples:
        ``` py title="Example"
        rogak_flagan_1992(1.5, 2.5)
        # 1.8027756377319946
        ```
    """
    return np.sqrt((fractal_dimension + 2) / 3) * radius_gyration


def zurita_gotor_2002(
    radius_gyration: Union[NDArray[np.float64], float],
    fractal_prefactor: Union[NDArray[np.float64], float],
) -> Union[NDArray[np.float64], float]:
    """Collision radius according to Zurita-Gotor and Rosner (2002).

    Args:
        radius_gyration: Radius of gyration of the particle [m].
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

    Examples:
        ``` py title="Example"
        zurita_gotor_2002(1.5, 1.2)
        # 1.568850650368
        ```
    """
    return 1.037 * (fractal_prefactor**0.077) * radius_gyration


def thajudeen_2012(
    fractal_dimension: float,
    number_of_particles: float,
    radius_gyration: Union[NDArray[np.float64], float],
    radius_monomer: float,
) -> Union[NDArray[np.float64], float]:
    """Collision radius according to Thajudeen et al. (2012).

    Args:
        fractal_dimension: Fractal dimension of the particle [dimensionless].
        number_of_particles: Number of particles in the aggregate.
        radius_gyration: Radius of gyration of the particle [m].
        radius_monomer: Monomer radius [m].

    Returns:
        (float): Collision radius of the particle [m].

    References:
        Thajudeen, T., Gopalakrishnan, R., & Hogan, C. J. (2012). The
        collision rate of nonspherical particles and aggregates for all
        diffusive knudsen numbers. Aerosol Science and Technology, 46(11).
        https://doi.org/10.1080/02786826.2012.701353

    Examples:
        ``` py title="Example"
        thajudeen_2012(2.5, 100, 1.5, 0.1)
        # 0.075
        ```
    """
    alpha1 = 0.253 * fractal_dimension**2 - 1.209 * fractal_dimension + 1.433
    alpha2 = -0.218 * fractal_dimension**2 + 0.964 * fractal_dimension - 0.180
    phi = 1 / (alpha1 * np.log(number_of_particles) + alpha2)
    radius_s_i = phi * radius_gyration
    radius_s_ii = (
        radius_monomer * (1.203 - 0.4315 / fractal_dimension) / 2
    ) * (4 * radius_s_i / radius_monomer) ** (
        0.8806 + 0.3497 / fractal_dimension
    )
    return radius_s_ii / 2


def qian_2022_rg(
    radius_gyration: Union[NDArray[np.float64], float],
    radius_monomer: float,
    coefficient: Tuple = (0.973, 0.441),
) -> Union[NDArray[np.float64], float]:
    """Fitted model using radius of gyration.

    Args:
        radius_gyration: radius of gyration [m].
        radius_monomer: monomer radius [m].

    Returns:
        (float): Collision radius of the particle [m].

    References:
        Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
        Effects of agglomerate characteristics on their collision kernels in
        the free molecular regime. Journal of Aerosol Science, 159.
        https://doi.org/10.1016/j.jaerosci.2021.105868

    Examples:
        ``` py title="Example"
        qian_2022_rg(1.5, 0.1)
        # 0.583
        ```
    """
    return (
        coefficient[0]
        * (radius_gyration / radius_monomer)
        + coefficient[1]
    ) * radius_monomer


def qian_2022_rg_df(
    fractal_dimension: Union[NDArray[np.float64], float],
    radius_gyration: Union[NDArray[np.float64], float],
    radius_monomer: float,
    coefficient: Tuple = (0.882, 0.223, 0.387),
) -> Union[NDArray[np.float64], float]:
    """New model Rc with Rg and Df parameters.

    Args:
        fractal_dimension: Fractal dimension of the particle [-].
        radius_gyration: Scaled radius of gyration [m].
        radius_monomer: Monomer radius [m].
        coefficient: Coefficients for the model

    Returns:
        (float): Collision radius of the particle [m].

    References:
        Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
        Effects of agglomerate characteristics on their collision kernels in
        the free molecular regime. Journal of Aerosol Science, 159.
        https://doi.org/10.1016/j.jaerosci.2021.105868

    Examples:
        ``` py title="Example"
        qian_2022_rg_df(2.5, 1.5, 0.1)
        # 0.583
        ```
    """
    return (
        coefficient[0]
        * (fractal_dimension**coefficient[1])
        * (radius_gyration / radius_monomer)
        + coefficient[2]
    ) * radius_monomer


def qian_2022_rg_df_k0(
    fractal_dimension: float,
    fractal_prefactor: float,
    radius_gyration: Union[NDArray[np.float64], float],
    radius_monomer: float,
    coefficient: Tuple = (0.777, 0.479, 0.000970, 0.267, -0.0790),
) -> Union[NDArray[np.float64], float]:
    """New model Rc with Rg, Df, and k0 parameters.

    Args:
        fractal_dimension: Fractal dimension of the particle [-].
        fractal_prefactor: Fractal prefactor of the particle [-].
        radius_gyration: radius of gyration [m].
        radius_monomer: monomer radius [m].
        coefficient: Coefficients for the model

    Returns:
        (float): Collision radius of the particle [m].

    References:
        Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
        Effects of agglomerate characteristics on their collision kernels in
        the free molecular regime. Journal of Aerosol Science, 159.
        https://doi.org/10.1016/j.jaerosci.2021.105868

    Examples:
        ``` py title="Example"
        qian_2022_rg_df_k0(2.5, 1.2, 1.5, 0.1)
        # 0.583
        ```
    """
    return (
        coefficient[0]
        * (fractal_dimension**coefficient[1])
        * (fractal_prefactor**coefficient[2])
        * (radius_gyration / radius_monomer)
        + coefficient[3] * fractal_prefactor
        + coefficient[4]
    ) * radius_monomer


# pylint: disable=too-many-arguments
def qian_2022_rg_df_k0_a13(
    fractal_dimension: float,
    fractal_prefactor: float,
    shape_anisotropy: float,
    radius_gyration: Union[NDArray[np.float64], float],
    radius_monomer: float,
    coefficient: Tuple = (0.876, 0.363, -0.105, 0.421, -0.0360, -0.227),
) -> Union[NDArray[np.float64], float]:
    """New model Rc with Rg, Df, k0, and A13 parameters.

    Args:
        fractal_dimension: Fractal dimension of the particle [-].
        fractal_prefactor: Fractal prefactor of the particle [-].
        shape_anisotropy: Parameter A13 for the model [-].
        radius_gyration: Radius of gyration [m].
        radius_monomer: Monomer radius [m].
        coefficient: Coefficients for the model

    Returns:
        (float): Collision radius of the particle [m].

    References:
        Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
        Effects of agglomerate characteristics on their collision kernels in
        the free molecular regime. Journal of Aerosol Science, 159.
        https://doi.org/10.1016/j.jaerosci.2021.105868

    Examples:
        ``` py title="Example"
        qian_2022_rg_df_k0_a13(2.5, 1.2, 0.5, 1.5, 0.1)
        # 0.583
        ```
    """
    return (
        coefficient[0]
        * (fractal_dimension**coefficient[1])
        * (fractal_prefactor**coefficient[2])
        * (radius_gyration / radius_monomer)
        + coefficient[3] * fractal_prefactor
        + coefficient[4] * shape_anisotropy
        + coefficient[5]
    ) * radius_monomer
