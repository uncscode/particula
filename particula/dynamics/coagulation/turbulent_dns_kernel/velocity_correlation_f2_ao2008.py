"""Velocity correlation terms for the two-point velocity correlation function
f₂(R) from Ayala et al. (2008).
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs

from .velocity_correlation_terms_ao2008 import compute_beta


@validate_inputs(
    {
        "collisional_radius": "positive",
        "taylor_microscale": "positive",
        "eulerian_integral_length": "positive",
    }
)
def get_f2_longitudinal_velocity_correlation(
    collisional_radius: Union[float, NDArray[np.float64]],
    taylor_microscale: Union[float, NDArray[np.float64]],
    eulerian_integral_length: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute the longitudinal velocity correlation function f₂(R) from
    Ayala et al. (2008).

    This function describes the correlation of velocity fluctuations as a
    function of collisional radius R between two colliding droplets.

    Where the equation is:

    - f₂(R) = 1 / (2√(1 - 2β²)) × {
        (1 + √(1 - 2β²)) exp[-2R / ((1 + √(1 - 2β²)) L_e)]
        - (1 - √(1 - 2β²)) exp[-2R / ((1 - √(1 - 2β²)) L_e)]
      }
        - f₂(R) is the longitudinal velocity correlation function [-].
        - R is the collisional radius [m].
        - β = (√2 * λ) / L_e
        - λ (taylor_microscale) : Taylor microscale [m].
        - L_e (eulerian_integral_length) : Eulerian integral length scale
          [m].

    Arguments:
        - collisional_radius : Distance between two colliding droplets [m].
        - taylor_microscale : Taylor microscale [m].
        - eulerian_integral_length : Eulerian integral length scale [m].

    Returns:
        - f₂(R) value [dimensionless].

    Examples:
        ```py
        import numpy as np
        example_f2 = get_f2_longitudinal_velocity_correlation(
            collisional_radius=np.array([1e-4, 2e-4]),
            taylor_microscale=1e-3,
            eulerian_integral_length=1e-2,
        )
        # Output: array([...])
        ```

    References:
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
      the geometric collision rate of sedimenting droplets. Part 2. Theory
      and parameterization. New Journal of Physics, 10.
      https://doi.org/10.1088/1367-2630/10/7/075016
    """
    beta = compute_beta(taylor_microscale, eulerian_integral_length)

    sqrt_term = np.sqrt(1 - 2 * beta**2)
    denominator = 2 * sqrt_term

    # Compute exponential terms
    exp_term_1 = np.exp(
        -2 * collisional_radius / ((1 + sqrt_term) * eulerian_integral_length)
    )
    exp_term_2 = np.exp(
        -2 * collisional_radius / ((1 - sqrt_term) * eulerian_integral_length)
    )

    # Compute f₂(R)
    return (1 / denominator) * (
        (1 + sqrt_term) * exp_term_1 - (1 - sqrt_term) * exp_term_2
    )
