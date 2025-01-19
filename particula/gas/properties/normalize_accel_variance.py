"""
Calculates the normalized acceleration variance in isotropic turbulence.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs


@validate_inputs({"re_lambda": "positive"})
def get_normalized_accel_variance(
    re_lambda: Union[float, NDArray[np.float64]],
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
    """
    return (11 + 7 * re_lambda) / (205 + re_lambda)
