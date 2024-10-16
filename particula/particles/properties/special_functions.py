"""
Special non-standard functions for aerosol properties
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray


def debye_function(
    variable: Union[float, NDArray[np.float64]],
    integration_points: int = 1000,
    n: int = 1,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Debye function for a given variable.

    The generalized Debye function where `x` is a dimensionless variable
    and `n` is an integer exponent. By default, `n` is 1,
    which corresponds to the most common form of the Debye function.

    Arguments:
        variable: The upper limit of integration. Can be a float or a
            numpy array. If a numpy array is provided, the function will
            return an array of Debye function values.
        integration_points: The number of points to use in the numerical
            integration. Default is 1000.
        n: The exponent in the Debye function. Default is 1.

    Returns:
        The value of the Debye function evaluation for the given input.
        If the input is a float, a float is returned. If the input is an array,
        an array of the same shape is returned.

    Examples:
        ``` py title="Dubye function with n=1 for a single float value"
        out = debye_function(1.0)
        print(out)
        # Output: 0.7765038970390566
        ```

        ``` py title="Dubye function with n=2 for a single float value"
        out = debye_function(1.0, n=2)
        print(out)
        # Output: 0.6007582206816492
        ```

        ``` py title="Dubye function with n=1 for a numpy array"
        out = debye_function(np.array([1.0, 2.0, 3.0]))
        print(out)
        # Output: [0.84140566 0.42278434 0.28784241]
        ```

    References:
        - https://en.wikipedia.org/wiki/Debye_function
        - https://mathworld.wolfram.com/DebyeFunctions.html
    """
    array = np.linspace(0, variable, integration_points)
    if n == 1:
        integral = np.trapz(
            array[1:] / (np.exp(array[1:]) - 1), array[1:], axis=0
        )
        return integral / variable

    integral = np.trapz(
        array[1:] ** n / (np.exp(array[1:]) - 1), array[1:], axis=0
    )
    return (n / variable**n) * integral
