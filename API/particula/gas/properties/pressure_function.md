# particula.gas.properties.pressure_function

*module*

Functions for calculating the partial pressure of a gas.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/pressure_function.py#L1-L1)

## particula.gas.properties.pressure_function.get_partial_pressure

*function*

```python
def get_partial_pressure(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate partial pressure from concentration, molar mass, T.

- p = (c × R × T) / M
    - p is Partial pressure [Pa].
    - c is Gas concentration [kg/m³].
    - R is Universal gas constant [J/(mol·K)].
    - T is Temperature [K].
    - M is Molar mass [kg/mol].

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas [kg/m³].
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas [kg/mol].
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature [K].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas [Pa].

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_partial_pressure(1.2, 0.02897, 298)
# Output: ~986.4 Pa
```

> **References**
>
> - Wikipedia contributors, "Ideal gas law," Wikipedia,
>   https://en.wikipedia.org/wiki/Ideal_gas_law

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/pressure_function.py#L12-L53)

## particula.gas.properties.pressure_function.get_saturation_ratio_from_pressure

*function*

```python
def get_saturation_ratio_from_pressure(partial_pressure: Union[float, NDArray[np.float64]], pure_vapor_pressure: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the saturation ratio of the gas at a given partial pressure and
pure vapor pressure.

The saturation ratio is defined as the ratio of partial pressure to the
pure vapor pressure.

- S = p / p_vap
    - S is Saturation ratio (dimensionless).
    - p is Partial pressure [Pa].
    - p_vap is Pure vapor pressure [Pa].

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Partial pressure [Pa].
- `pure_vapor_pressure` (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Saturation ratio of the gas (dimensionless).

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_saturation_ratio_from_pressure(800.0, 1000.0)
# Output: 0.8
```

> **References**
>
> - Wikipedia contributors, "Relative humidity," Wikipedia,
>   https://en.wikipedia.org/wiki/Relative_humidity

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/pressure_function.py#L56-L95)
