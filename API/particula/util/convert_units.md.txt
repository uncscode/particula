# particula.util.convert_units

*module*

This module provides a Pint-based wrapper to convert values between different
units. The Pint library must be installed to use all features.

**Examples**

``` py title="Basic Usage"
from particula.util.convert_units import get_unit_conversion

# Convert 10 degrees Celsius to degrees Fahrenheit:
converted_value = get_unit_conversion("degC", "degF", value=10)
print(converted_value)
# ~50.0
```

> **References**
>
> - Pint documentation: https://pint.readthedocs.io/

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_units.py#L1-L1)

## particula.util.convert_units.ReturnType

*attribute* · *module attribute*

```python
ReturnType = Union[float, np.ndarray]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_units.py#L30-L30)

## particula.util.convert_units.unit_registry

*attribute* · *module attribute*

```python
unit_registry: Optional[pint.UnitRegistry] = pint.UnitRegistry()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_units.py#L25-L25)

## particula.util.convert_units.get_unit_conversion

*function*

```python
def get_unit_conversion(old: str, new: str, value: Optional[Union[float, np.ndarray]] = None) -> ReturnType
```

Convert a numeric value or unit expression from one unit to another using
Pint.

For simple multiplicative units, if no value is provided, this function
returns the conversion factor. For units with an offset
(e.g., temperatures), or if a value is supplied, a fully converted
numeric value is returned instead.

**Parameters**

- `old` (`str`) — A string representing the current unit (e.g., "m", "degC").
- `new` (`str`) — A string representing the target unit.
- `value` (`Optional[Union[float, np.ndarray]]`) (default: `None`) — An optional numeric value to convert. If omitted, returns the conversion factor between old and new.

**Raises**

- `ImportError` — If Pint is not installed. Install it using:
`pip install pint`.

**Returns**

- (`ReturnType`) — A float or ``numpy.ndarray`` representing either the conversion factor or the fully converted value in the target unit.

**Examples**

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

> **References**
>
> - Pint documentation: https://pint.readthedocs.io/

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_units.py#L33-L99)
