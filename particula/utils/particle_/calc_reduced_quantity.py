""" Calculate any reduced quantity:

    reduced_quantity =
        quantity_1 * quantity_2 / (quantity_1 + quantity_2)
"""

from particula import u


def reduced_quantity(a_quantity, b_quantity) -> float:

    """ Returns the reduced mass of two particles.

        Parameters:
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

    return a_quantity * b_quantity / (a_quantity + b_quantity)
