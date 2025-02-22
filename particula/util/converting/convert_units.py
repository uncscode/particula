"""
This module provides a Pint-based wrapper to convert values between different
units. The Pint library must be installed to use all features.

Examples:
    ``` py title="Basic Usage"
    from particula.util.converting.convert_units import get_unit_conversion

    # Convert 10 degrees Celsius to degrees Fahrenheit:
    converted_value = get_unit_conversion("degC", "degF", value=10)
    print(converted_value)
    # ~50.0
    ```

References:
    - Pint documentation: https://pint.readthedocs.io/
"""

from typing import Optional

try:
    import pint

    UNIT_REGISTRY = pint.UnitRegistry()
except ImportError:
    UNIT_REGISTRY = None


def get_unit_conversion(
    old: str,
    new: str,
    value: Optional[float] = None,
) -> float:
    """
    Convert a numeric value or unit expression from one unit to another using
    Pint.

    For simple multiplicative units, if no value is provided, this function
    returns the conversion factor. For units with an offset
    (e.g., temperatures), or if a value is supplied, a fully converted
    numeric value is returned instead.

    Arguments:
        - old : A string representing the current unit (e.g., "m", "degC").
        - new : A string representing the target unit.
        - value : An optional numeric value to convert. If omitted, returns the
            conversion factor between old and new.

    Raises:
        - ImportError : If Pint is not installed. Install it using:
            `pip install pint`.

    Returns:
        - A float representing either the conversion factor or the fully
          converted value in the target unit.

    Examples:
        ``` py title="Example Multi-Unit Conversion"
        import particula as par
        factor = par.get_unit_conversion("ug/m^3", "kg/m^3")
        print(factor)
        # 1e-9
        ```

        ``` py title="Example Temperature Conversion"
        import particula as par
        degF = par.get_unit_conversion("degC", "degF", value=25)
        print(degF)
        # ~77.0
        ```

    References:
        - Pint documentation: https://pint.readthedocs.io/
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
