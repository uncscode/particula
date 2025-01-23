"""
Calculates the normalized acceleration variance in isotropic turbulence.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs


@validate_inputs({"re_lambda": "positive"})
def get_normalized_accel_variance_ao2008(
    re_lambda: Union[float, NDArray[np.float64]],
    numerical_stability_epsilon: float = 1e-14,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the normalized acceleration variance in isotropic turbulence.

    This coefficient describes the statistical behavior of acceleration
    fluctuations in turbulent flows. It is given by:

        a_o = (11 + 7 R_λ) / (205 + R_λ)

    - a_o (accel_variance) : Normalized acceleration variance in isotropic
        turbulence [-]
    - R_λ (re_lambda) : Taylor-microscale Reynolds number [-]

    Arguments:
    ----------
        - re_lambda : Taylor-microscale Reynolds number [-]

    Returns:
    --------
        - accel_variance : Normalized acceleration variance [-]

    References:
    -----------
    - The equivalent numerically stable version used is this.
        (7 + 11 / (R_λ + ε)) / (1 + 205 / (R_λ + ε))
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2. Theory
        and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    return (7 + 11 / (re_lambda + numerical_stability_epsilon)) / (
        1 + 205 / (re_lambda + numerical_stability_epsilon)
    )
