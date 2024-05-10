""" calculating reduced quantity.

    reduced_quantity =
        quantity_1 * quantity_2 / (quantity_1 + quantity_2)
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula import u


def reduced_quantity(a_quantity, b_quantity):
    """ Returns the reduced mass of two particles.

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
    alpha: Union[float, NDArray[np.float_]],
    beta: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]:
    """
    Returns the reduced value of two parameters.
    reduced_value = alpha * beta / (alpha + beta)

    Args:
    - alpha: The first parameter.
    - beta: The second parameter.

    Returns:
    -------
    - Same dimension as the input parameters.

    A reduced quantity is an "effective inertial" quantity,
    allowing two-body problems to be solved as one-body problems.
    """
    # If they are arrays check the shapes are identical
    if isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray):
        if alpha.shape != beta.shape:
            raise ValueError(
                "The shapes of the input arrays must be identical."
            )
    return alpha * beta / (alpha + beta)
