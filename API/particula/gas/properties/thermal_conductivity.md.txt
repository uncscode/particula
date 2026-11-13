# particula.gas.properties.thermal_conductivity

*module*

Thermal Conductivity of air.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/thermal_conductivity.py#L1-L1)

## particula.gas.properties.thermal_conductivity.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/thermal_conductivity.py#L11-L11)

## particula.gas.properties.thermal_conductivity.get_thermal_conductivity

*function*

```python
def get_thermal_conductivity(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Thermal conductivity of air as a function of temperature.

Calculate the thermal conductivity of air as a function of temperature.
Based on a simplified linear relation from atmospheric science literature.
Only valid for temperatures within the range typically found on
Earth's surface.

- k(T) = 1e-3 × (4.39 + 0.071 × T)
    - k(T) is Thermal conductivity [W/(m·K)].
    - T is Temperature [K].

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — The temperature in Kelvin (K).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The thermal conductivity [W/(m·K)] or [J/(m·s·K)].

**Examples**

``` py title="Example Usage"
import particula as par
par.gas.get_thermal_conductivity(300)
# ~0.449 W/(m·K)
```

References:
- Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 17.54.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/thermal_conductivity.py#L14-L45)
