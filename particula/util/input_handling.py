""" handling inputs
"""

from typing import Union, Optional
from particula import u


def convert_units(
    old: Union[str, u.Quantity],
    new: Union[str, u.Quantity],
    value: Optional[float] = None
) -> float:
    """ generic pint function to convert units

        Args:
            old     [str | u.Quantity]
            new     [str | u.Quantity]
            value   (float) [optional]

        Returns:
            multiplier     (float)

        Notes:
            * If unit is correct, take to base units
            * Throws ValueError if unit is wrong
            * Assigning default base units to scalar input
    """
    if isinstance(old, str):
        addative_units = ['degC', 'degF', 'degR', 'degK']
        if old in addative_units or value is not None:
            value = value if value is not None else 0
            old = u.Quantity(value, old)
        else:
            old = 1 * u.Quantity(old)
    if isinstance(new, str):
        new = u.Quantity(new)

    if isinstance(old, u.Quantity) and isinstance(new, u.Quantity):
        new_value = old.to(new)
    else:
        raise ValueError(
            f"\n\t"
            f"Input has unsupported units.\n\t"
            f"Input must have units equivlanet to {new};\n\t"
            f"otherwise, if dimensionless, it will\n\t"
            f"be assigned {new}.\n"
        )
    return float(new_value.m)
