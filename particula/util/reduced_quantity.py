"""
This module provides functions to calculate a reduced quantity between
parameters or across an array, useful in multi-body or multi-parameter
problems.
"""

import logging
from typing import Union
from numpy.typing import NDArray
import numpy as np

logger = logging.getLogger("particula")  # get instance of logger


def get_reduced_value(
    alpha: Union[float, NDArray[np.float64]],
    beta: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Return the reduced value of two parameters.

    The reduced value is computed using:
    - r = (α × β) / (α + β),
        - r is the reduced quantity,
        - α, β are the input parameters.

    Arguments:
        - alpha : The first parameter (scalar or array).
        - beta : The second parameter (scalar or array).

    Returns:
        - The element-wise reduced quantity, zero if (α+β)=0.

    Raises:
        - ValueError : If arrays have incompatible shapes.

    Examples:
        ``` py title="Example"
        from particula.util.reduced_quantity import get_reduced_value
        import numpy as np

        print(get_reduced_value(3.0, 6.0))
        # Output: 2.0

        arrA = np.array([1.0, 2.0, 3.0])
        arrB = np.array([2.0, 5.0, 10.0])
        print(get_reduced_value(arrA, arrB))
        # Output: [0.666..., 1.428..., 2.142...]
        ```

    References:
        - [Reduced Mass, Wikipedia](https://en.wikipedia.org/wiki/Reduced_mass)
    """
    # Ensure input compatibility, especially when both are arrays
    if (
        isinstance(alpha, np.ndarray)
        and isinstance(beta, np.ndarray)
        and (alpha.shape != beta.shape)
    ):
        logger.error("The shapes of alpha and beta must be identical.")
        raise ValueError("The shapes of alpha and beta must be identical.")

    # Calculation of the reduced value, with safety against division by zero
    denominator = alpha + beta
    # Using np.errstate to suppress divide by zero warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, alpha * beta / denominator, 0)
    return result


def get_reduced_self_broadcast(
    alpha_array: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Return a square matrix of pairwise reduced values using a single array.

    Each element is calculated by broadcasting the array with its transpose:
    - r_ij = (α_i × α_j) / (α_i + α_j),
        - r_ij is the reduced quantity between α_i and α_j.

    Arguments:
        - alpha_array : A 1D array for pairwise reduced value calculations.

    Returns:
        - A 2D square matrix of pairwise reduced values.

    Examples:
        ``` py title="Example"
        from particula.util.reduced_quantity import get_reduced_self_broadcast
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0])
        print(get_reduced_self_broadcast(arr))
        # Output: [[0.5       0.6666667 0.75     ]
        #          [0.6666667 1.        1.2      ]
        #          [0.75      1.2       1.5      ]]
        ```

    References:
        - [Reduced Mass, Wikipedia](https://en.wikipedia.org/wiki/Reduced_mass)
    """
    # Use broadcasting to create matrix and its transpose
    alpha_matrix = alpha_array[:, np.newaxis]
    alpha_matrix_transpose = alpha_array[np.newaxis, :]
    denominator = alpha_matrix + alpha_matrix_transpose
    # Perform element-wise multiplication and division with np.errstate
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            denominator != 0,
            alpha_matrix * alpha_matrix_transpose / denominator,
            0,
        )
    return result
