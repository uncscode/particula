"""Special non-standard functions for aerosol properties."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.machine_limit import get_safe_exp
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "variable": "finite",
    }
)
def get_debye_function(
    variable: Union[float, NDArray[np.float64]],
    integration_points: int = 1000,
    n: int = 1,
) -> Union[float, NDArray[np.float64]]:
    """Calculate the generalized Debye function for a given input.

    The Debye function can be expressed as follows:

    - Dₙ(x) = (n / xⁿ) ∫[tⁿ / (exp(t) - 1)] dt  from t = 0 to x
        - x is a dimensionless variable.
        - n is the exponent (default is 1).

    Arguments:
        - variable : Upper limit of integration; can be float or NDArray.
        - integration_points : Number of points for numerical integration
          (default 1000).
        - n : Exponent in the Debye function formula (default 1).

    Returns:
        - Debye function value(s). If the input is a float, returns a float.
          If the input is an array, returns an array of the same shape.

    Examples:
        ``` py title="Debye function with n=1 for a single float value"
        import particula as par
        par.particles.get_debye_function(1.0)
        # Output: 0.7765038970390566
        ```

        ``` py title="Debye function with n=2 for a single float value"
        import particula as par
        par.particles.get_debye_function(1.0, n=2)
        # Output: 0.6007582206816492
        ```

        ``` py title="Debye function with n=1 for a numpy array"
        import particula as par
        par.particles.get_debye_function(np.array([1.0, 2.0, 3.0]))
        # Output: [0.84140566 0.42278434 0.28784241]
        ```

    References:
        - [Debye function](https://en.wikipedia.org/wiki/Debye_function)
        - [Wolfram MathWorld: Debye Functions](https://mathworld.wolfram.com/DebyeFunctions.html)
    """
    array = np.linspace(0, variable, integration_points)
    exp_array = get_safe_exp(array[1:])

    if n == 1:
        integral = np.trapezoid(array[1:] / (exp_array - 1), array[1:], axis=0)
        return integral / variable

    integral = np.trapezoid(array[1:] ** n / (exp_array - 1), array[1:], axis=0)
    return (n / variable**n) * integral
