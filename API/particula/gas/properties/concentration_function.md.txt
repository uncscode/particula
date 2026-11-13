# particula.gas.properties.concentration_function

*module*

Function for calculating the gas concentrations.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/concentration_function.py#L1-L1)

## particula.gas.properties.concentration_function.get_concentration_from_pressure

*function*

```python
def get_concentration_from_pressure(partial_pressure: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the concentration of a gas using the ideal gas law.

The concentration is determined from the partial pressure, molar mass,
and temperature using the ideal gas equation:

- C = (P × M) / (R × T)
    - C is the concentration in kg/m³,
    - P is the partial pressure in Pascals (Pa),
    - M is the molar mass in kg/mol,
    - R is the universal gas constant (J/(mol·K)),
    - T is the temperature in Kelvin.

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas in Pascals (Pa).
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Examples**

```py title="Floating-point Example Usage"
import particula as par
par.gas.get_concentration_from_pressure(101325, 0.02897, 298.15)
# Output: 1.184587604735883
```

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m³.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/concentration_function.py#L11-L43)
