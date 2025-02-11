"""
Pint Unit Conversion Wrapper

This is the only module that uses the pint library.
"""

from typing import Optional

try:
    import pint

    UNIT_REGISTRY = pint.UnitRegistry()
except ImportError:
    UNIT_REGISTRY = None


def convert_units(
    old: str,
    new: str,
    value: Optional[float] = None,
) -> float:
    """Generic wrapper for pint to convert units

    Args:
        - old : old units defined by pint, e.g., "m", "ft", "Km", "kg/m^3"
        - new : new units defined by pint, e.g., "m", "ft", "Km", "kg/m^3"
        - value : value to convert, needed for units with and offset
            e.g., "degC"

    Returns:
        - float : conversion multiplier from old to new units. If value is
            provided, it returns the converted value in the new units.

    Raises:
        ImportError: if pint is not installed

    Examples:
    ``` py title="Example Usage"
    conversion_multipliter = convert_units("ug/m^3", "kg/m^3")
    # 1e-9
    ```
    """
    if UNIT_REGISTRY is None:
        raise ImportError(
            "Install pint to use unit conversion features: pip install pint"
        )

    offset_units = ["degC", "degF", "degR", "degK"]
    if old in offset_units or value is not None:
        value = value if value is not None else 0
        old = UNIT_REGISTRY.Quantity(value, old)
    else:
        old = UNIT_REGISTRY.Quantity(old)  # multiplicative shift

    new = UNIT_REGISTRY.Quantity(new)
    result = old.to(new).magnitude  # get the new value without units
    return float(result)
