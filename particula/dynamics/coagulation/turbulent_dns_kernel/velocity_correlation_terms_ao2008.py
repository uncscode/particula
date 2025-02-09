"""
Velocity correlation terms for the DNS kernel of the turbulent coagulation
model by Ayala 2008.

- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2.
    Theory and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "lagrangian_taylor_microscale_time": "positive",
        "lagrangian_integral_scale": "positive",
    }
)
def compute_z(
    lagrangian_taylor_microscale_time: Union[float, NDArray[np.float64]],
    lagrangian_integral_scale: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute z, which is defined as:

        z = τ_T / T_L L_e

    - τ_T (lagrangian_taylor_microscale_time) : Lagrangian Taylor
        microscale time [s].
    - L_e (Lagrangian integral scale) : Lagrangian integral scale [s].

    Arguments:
    ----------
        - lagrangian_taylor_microscale_time : Lagrangian Taylor microscale
            time [s].
        - Lagrangian integral scale : Lagrangian integral scale [s].

    Returns:
    --------
        - z value [-].
    """
    return lagrangian_taylor_microscale_time / lagrangian_integral_scale


@validate_inputs(
    {
        "taylor_microscale": "positive",
        "eulerian_integral_length": "positive",
    }
)
def compute_beta(
    taylor_microscale: Union[float, NDArray[np.float64]],
    eulerian_integral_length: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute β, which is defined as:

        β = (sqrt(2) * λ) / L_e

    - λ (taylor_microscale) : Taylor microscale [m].
    - L_e (eulerian_integral_length) : Eulerian integral length scale [m].

    Arguments:
    ----------
        - taylor_microscale : Taylor microscale [m].
        - eulerian_integral_length : Eulerian integral length scale [m].

    Returns:
    --------
        - β value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return (np.sqrt(2) * taylor_microscale) / eulerian_integral_length


@validate_inputs({"z": "positive"})
def compute_b1(
    z: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """
    Compute b₁, which is defined as:

        b₁ = (1 + sqrt(1 - 2z²)) / (2 sqrt(1 - 2z²))

    - z : Defined as z = τ_T / L_e.

    Arguments:
    ----------
        - z : A dimensionless parameter related to turbulence [-].

    Returns:
    --------
        - b₁ value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    sqrt_term = np.sqrt(1 - 2 * z**2)
    return (1 + sqrt_term) / (2 * sqrt_term)


@validate_inputs({"z": "positive"})
def compute_b2(
    z: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """
    Compute b₂, which is defined as:

        b₂ = (1 - sqrt(1 - 2z²)) / (2 sqrt(1 - 2z²))

    - z : Defined as z = τ_T / L_e.

    Arguments:
    ----------
        - z : A dimensionless parameter related to turbulence [-].

    Returns:
    --------
        - b₂ value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    sqrt_term = np.sqrt(1 - 2 * z**2)
    return (1 - sqrt_term) / (2 * sqrt_term)


@validate_inputs({"z": "positive", "lagrangian_integral_time": "positive"})
def compute_c1(
    z: Union[float, NDArray[np.float64]],
    lagrangian_integral_time: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute c₁, which is defined as:

        c₁ = ((1 + sqrt(1 - 2z²)) * T_L) / 2

    - z : Defined as z = τ_T / L_e.
    - T_L (lagrangian_integral_time) : Lagrangian integral timescale [s].

    Arguments:
    ----------
        - z : A dimensionless parameter related to turbulence [-].
        - lagrangian_integral_time : Lagrangian integral timescale [s].

    Returns:
    --------
        - c₁ value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return ((1 + np.sqrt(1 - 2 * z**2)) * lagrangian_integral_time) / 2


@validate_inputs({"z": "positive", "lagrangian_integral_time": "positive"})
def compute_c2(
    z: Union[float, NDArray[np.float64]],
    lagrangian_integral_time: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute c₂, which is defined as:

        c₂ = ((1 - sqrt(1 - 2z²)) * T_L) / 2

    - z : Defined as z = τ_T / L_e.
    - T_L (lagrangian_integral_time) : Lagrangian integral timescale [s].

    Arguments:
    ----------
        - z : A dimensionless parameter related to turbulence [-].
        - lagrangian_integral_time : Lagrangian integral timescale [s].

    Returns:
    --------
        - c₂ value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return ((1 - np.sqrt(1 - 2 * z**2)) * lagrangian_integral_time) / 2


@validate_inputs({"beta": "positive"})
def compute_d1(
    beta: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """
    Compute d₁, which is defined as:

        d₁ = (1 + sqrt(1 - 2β²)) / (2 sqrt(1 - 2β²))

    - β : Defined as β = (sqrt(2) * λ) / L_e.

    Arguments:
    ----------
        - beta : A dimensionless parameter related to turbulence [-].

    Returns:
    --------
        - d₁ value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    sqrt_term = np.sqrt(1 - 2 * beta**2)
    return (1 + sqrt_term) / (2 * sqrt_term)


@validate_inputs({"beta": "positive"})
def compute_d2(
    beta: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """
    Compute d₂, which is defined as:

        d₂ = (1 - sqrt(1 - 2β²)) / (2 sqrt(1 - 2β²))

    - β : Defined as β = (sqrt(2) * λ) / L_e.

    Arguments:
    ----------
        - beta : A dimensionless parameter related to turbulence [-].

    Returns:
    --------
        - d₂ value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    sqrt_term = np.sqrt(1 - 2 * beta**2)
    return (1 - sqrt_term) / (2 * sqrt_term)


@validate_inputs({"beta": "positive", "eulerian_integral_length": "positive"})
def compute_e1(
    beta: Union[float, NDArray[np.float64]],
    eulerian_integral_length: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute e₁, which is defined as:

        e₁ = ((1 + sqrt(1 - 2β²)) * L_e) / 2

    - β : Defined as β = (sqrt(2) * λ) / L_e.
    - L_e (eulerian_integral_length) : Eulerian integral length scale [m].

    Arguments:
    ----------
        - beta : A dimensionless parameter related to turbulence [-].
        - eulerian_integral_length : Eulerian integral length scale [m].

    Returns:
    --------
        - e₁ value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return ((1 + np.sqrt(1 - 2 * beta**2)) * eulerian_integral_length) / 2


@validate_inputs({"beta": "positive", "eulerian_integral_length": "positive"})
def compute_e2(
    beta: Union[float, NDArray[np.float64]],
    eulerian_integral_length: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute e₂, which is defined as:

        e₂ = ((1 - sqrt(1 - 2β²)) * L_e) / 2

    - β : Defined as β = (sqrt(2) * λ) / L_e.
    - L_e (eulerian_integral_length) : Eulerian integral length scale [m].

    Arguments:
    ----------
        - beta : A dimensionless parameter related to turbulence [-].
        - eulerian_integral_length : Eulerian integral length scale [m].

    Returns:
    --------
        - e₂ value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return ((1 - np.sqrt(1 - 2 * beta**2)) * eulerian_integral_length) / 2
