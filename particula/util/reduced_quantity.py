""" calculating reduced quantity.

    reduced_quantity =
        quantity_1 * quantity_2 / (quantity_1 + quantity_2)
"""

import logging
from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula import u

logger = logging.getLogger("particula")  # get instance of logger


def reduced_quantity(a_quantity, b_quantity):
    """Returns the reduced mass of two particles.

    Examples:
    ```
    >>> reduced_quantity(1*u.kg, 1*u.kg)
    <Quantity(0.5, 'kilogram')>
    >>> reduced_quantity(1*u.kg, 20*u.kg).m
    0.9523809523809523
    >>> reduced_quantity(1, 200)
    0.9950248756218906
    >>> reduced_quantity([1, 2, 3], 200)
    array([0.99502488, 1.98019802, 2.95566502])
    >>> reduced_quantity([1, 2], [200, 300])
    array([0.99502488, 1.98675497])
    ```

    Args:
        a_quantity  (float)  [arbitrary units]
        b_quantity  (float)  [arbitrary units]

    Returns:
                    (float)  [arbitrary units]

    A reduced quantity is an "effective inertial" quantity,
    allowing two-body problems to be solved as one-body problems.
    """

    a_q = a_quantity
    b_q = b_quantity

    if isinstance(a_q, u.Quantity):
        a_q = a_q.to_base_units()
        if not isinstance(b_q, u.Quantity):
            raise TypeError(
                f"\n\t"
                f"{a_q} and {b_q} (dimensionless) not compatible!\n\t"
                f"Quantities must have same units to be reduced.\n\t"
                f"Try: {a_q} and {b_q} {a_q.units} for example.\n"
            )
        if a_q.units != b_q.to_base_units().units:
            raise TypeError(
                f"\n\t"
                f"{a_q} and {b_q} not compatible!\n"
                f"Quantities must have same units to be reduced.\n\t"
                f"Try: {a_q} and {b_q} {a_q.units} for example."
            )
    elif isinstance(b_q, u.Quantity):
        b_q = b_q.to_base_units()
        if not isinstance(a_q, u.Quantity):
            raise TypeError(
                f"\n\t"
                f"{a_q} (dimensionless) and {b_q} not compatible!\n\t"
                f"Quantities must have same units to be reduced.\n\t"
                f"Try: {b_q} and {a_q} {b_q.units} for example."
            )
        if a_q.to_base_units().units != b_q.units:
            raise TypeError(
                f"\n\t"
                f"{a_q} and {b_q} not compatible!\n\t"
                f"Quantities must have same units to be reduced.\n\t"
                f"Try: {b_q} and {a_q} {b_q.units} for example"
            )

    if not isinstance(a_q, u.Quantity):
        a_q = u.Quantity(a_q, " ")
    if not isinstance(b_q, u.Quantity):
        b_q = u.Quantity(b_q, " ")

    return (a_q * b_q / (a_q + b_q)).to_base_units()


def reduced_value(
    alpha: Union[float, NDArray[np.float64]],
    beta: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Returns the reduced value of two parameters, calculated as:
    reduced_value = alpha * beta / (alpha + beta)

    This formula calculates an "effective inertial" quantity,
    allowing two-body problems to be solved as if they were one-body problems.

    Args:
    - alpha: The first parameter (scalar or array).
    - beta: The second parameter (scalar or array).

    Returns:
    -------
    - A value or array of the same dimension as the input parameters. Returns
      zero where alpha + beta equals zero to handle division by zero
      gracefully.

    Raises:
    - ValueError: If alpha and beta are arrays and their shapes do not match.
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
    # Using np.where to avoid division by zero
    return np.where(denominator != 0, alpha * beta / denominator, 0)


def reduced_self_broadcast(
    alpha_array: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Returns the reduced value of an array with itself, broadcasting the
    array into a matrix and calculating the reduced value of each element pair.
    reduced_value = alpha_matrix * alpha_matrix_Transpose
                    / (alpha_matrix + alpha_matrix_Transpose)

    Args:
    - alpha_array: The array to be broadcast and reduced.

    Returns:
    -------
    - A square matrix of the reduced values.
    """
    # Use broadcasting to create matrix and its transpose
    alpha_matrix = alpha_array[:, np.newaxis]
    alpha_matrix_transpose = alpha_array[np.newaxis, :]
    denominator = alpha_matrix + alpha_matrix_transpose
    # Perform element-wise multiplication and division
    return np.where(
        denominator != 0,
        alpha_matrix * alpha_matrix_transpose / denominator,
        0,
    )
