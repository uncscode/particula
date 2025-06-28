"""Velocity correlation terms for the DNS kernel of the turbulent coagulation
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
    """Compute z, the ratio of Taylor microscale time to Lagrangian timescale.

    Where the equation is
    - z = τ_T / T_L
        - τ_T (lagrangian_taylor_microscale_time) is the Lagrangian Taylor
            microscale time [s].
        - T_L (lagrangian_integral_scale) is the Lagrangian integral
            timescale [s].

    Arguments:
        - lagrangian_taylor_microscale_time : Lagrangian Taylor microscale
            time [s].
        - lagrangian_integral_scale : Lagrangian integral timescale [s].

    Returns:
        - z value [dimensionless].

    Examples:
        ```py
        example_z = compute_z(0.5, 1.0)
        # Output: 0.5
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
            the geometric collision rate of sedimenting droplets. Part 2.
            Theory and parameterization. New Journal of Physics, 10.
            https://doi.org/10.1088/1367-2630/10/7/075016
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
    """Compute β, the ratio of microscale to integral length scale.

    - β = (√2 × λ) / L_e
        - λ is Taylor microscale [m].
        - L_e is Eulerian integral length scale [m].

    Arguments:
        - taylor_microscale : Taylor microscale [m].
        - eulerian_integral_length : Eulerian integral length scale [m].

    Returns:
        - β value [dimensionless].

    Examples:
        ```py
        beta_val = compute_beta(0.001, 0.1)
        # Output: 0.01414
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
            the geometric collision rate of sedimenting droplets. Part 2.
            Theory and parameterization. New Journal of Physics, 10.
            https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return (np.sqrt(2) * taylor_microscale) / eulerian_integral_length


@validate_inputs({"z": "positive"})
def compute_b1(
    z: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute b₁, a dimensionless parameter in the Ayala 2008 model.

    - b₁ = (1 + √(1 - 2z²)) / (2 √(1 - 2z²))
        - z is τ_T / T_L.

    Arguments:
        - z : A dimensionless parameter related to turbulence [-].

    Returns:
        - b₁ value [dimensionless].

    Examples:
        ```py
        b1_val = compute_b1(0.5)
        # Output: 0.866
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
            the geometric collision rate of sedimenting droplets. Part 2.
            Theory and parameterization. New Journal of Physics, 10.
            https://doi.org/10.1088/1367-2630/10/7/075016
    """
    sqrt_term = np.sqrt(1 - 2 * z**2)
    return (1 + sqrt_term) / (2 * sqrt_term)


@validate_inputs({"z": "positive"})
def compute_b2(
    z: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute b₂, a dimensionless parameter in the Ayala 2008 model.

    - b₂ = (1 - √(1 - 2z²)) / (2 √(1 - 2z²))
        - z is τ_T / T_L.

    Arguments:
        - z : A dimensionless parameter related to turbulence [-].

    Returns:
        - b₂ value [dimensionless].

    Examples:
        ```py
        b2_val = compute_b2(0.5)
        # Output: 0.134
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
            the geometric collision rate of sedimenting droplets. Part 2.
            Theory and parameterization. New Journal of Physics, 10.
            https://doi.org/10.1088/1367-2630/10/7/075016
    """
    sqrt_term = np.sqrt(1 - 2 * z**2)
    return (1 - sqrt_term) / (2 * sqrt_term)


@validate_inputs({"z": "positive", "lagrangian_integral_scale": "positive"})
def compute_c1(
    z: Union[float, NDArray[np.float64]],
    lagrangian_integral_scale: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute c₁, a dimensionless timescale factor in the Ayala 2008 model.

    - c₁ = ((1 + √(1 - 2z²)) × T_L) / 2
        - z is τ_T / T_L.
        - T_L is the Lagrangian integral timescale [s].

    Arguments:
        - z : A dimensionless parameter related to turbulence [-].
        - lagrangian_integral_scale : Lagrangian integral timescale [s].

    Returns:
        - c₁ value [dimensionless].

    Examples:
        ```py
        c1_val = compute_c1(0.5, 1.0)
        # Output: 0.933
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
            the geometric collision rate of sedimenting droplets. Part 2.
            Theory and parameterization. New Journal of Physics, 10.
            https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return ((1 + np.sqrt(1 - 2 * z**2)) * lagrangian_integral_scale) / 2


@validate_inputs({"z": "positive", "lagrangian_integral_scale": "positive"})
def compute_c2(
    z: Union[float, NDArray[np.float64]],
    lagrangian_integral_scale: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute c₂, a dimensionless timescale factor in the Ayala 2008 model.

    - c₂ = ((1 - √(1 - 2z²)) × T_L) / 2
        - z is τ_T / T_L.
        - T_L is the Lagrangian integral timescale [s].

    Arguments:
        - z : A dimensionless parameter related to turbulence [-].
        - lagrangian_integral_scale : Lagrangian integral timescale [s].

    Returns:
        - c₂ value [dimensionless].

    Examples:
        ```py
        c2_val = compute_c2(0.5, 1.0)
        # Output: 0.067
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
            the geometric collision rate of sedimenting droplets. Part 2.
            Theory and parameterization. New Journal of Physics, 10.
            https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return ((1 - np.sqrt(1 - 2 * z**2)) * lagrangian_integral_scale) / 2


@validate_inputs({"beta": "positive"})
def compute_d1(
    beta: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute d₁, another dimensionless coefficient from Ayala 2008.

    - d₁ = (1 + √(1 - 2β²)) / (2 √(1 - 2β²))
        - β is defined as β = (√2 × λ) / L_e.

    Arguments:
        - beta : A dimensionless parameter related to turbulence [-].

    Returns:
        - d₁ value [dimensionless].

    Examples:
        ```py
        d1_val = compute_d1(0.5)
        # Output: 0.866
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
            the geometric collision rate of sedimenting droplets. Part 2.
            Theory and parameterization. New Journal of Physics, 10.
            https://doi.org/10.1088/1367-2630/10/7/075016
    """
    sqrt_term = np.sqrt(1 - 2 * beta**2)
    return (1 + sqrt_term) / (2 * sqrt_term)


@validate_inputs({"beta": "positive"})
def compute_d2(
    beta: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute d₂, another dimensionless coefficient from Ayala 2008.

    - d₂ = (1 - √(1 - 2β²)) / (2 √(1 - 2β²))
        - β is defined as β = (√2 × λ) / L_e.

    Arguments:
        - beta : A dimensionless parameter related to turbulence [-].

    Returns:
        - d₂ value [dimensionless].

    Examples:
        ```py
        d2_val = compute_d2(0.5)
        # Output: 0.134
        ```

    References:
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
    """Compute e₁, which scales the integral length by a factor in Ayala 2008.

    - e₁ = ((1 + √(1 - 2β²)) × L_e) / 2
        - β is defined as β = (√2 × λ) / L_e.
        - L_e is the Eulerian integral length scale [m].

    Arguments:
        - beta : A dimensionless parameter related to turbulence [-].
        - eulerian_integral_length : Eulerian integral length scale [m].

    Returns:
        - e₁ value [dimensionless].

    Examples:
        ```py
        e1_val = compute_e1(0.5, 0.1)
        # Output: 0.0866
        ```

    References:
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
    """Compute e₂, which scales the integral length by a factor in Ayala 2008.

    - e₂ = ((1 - √(1 - 2β²)) × L_e) / 2
        - β is defined as β = (√2 × λ) / L_e.
        - L_e is the Eulerian integral length scale [m].

    Arguments:
        - beta : A dimensionless parameter related to turbulence [-].
        - eulerian_integral_length : Eulerian integral length scale [m].

    Returns:
        - e₂ value [dimensionless].

    Examples:
        ```py
        e2_val = compute_e2(0.5, 0.1)
        # Output: 0.0134
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
            the geometric collision rate of sedimenting droplets. Part 2.
            Theory and parameterization. New Journal of Physics, 10.
            https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return ((1 - np.sqrt(1 - 2 * beta**2)) * eulerian_integral_length) / 2
