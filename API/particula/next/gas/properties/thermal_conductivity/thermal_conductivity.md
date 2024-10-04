# Thermal Conductivity

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Thermal Conductivity

> Auto-generated documentation for [particula.next.gas.properties.thermal_conductivity](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/thermal_conductivity.py) module.

## get_thermal_conductivity

[Show source in thermal_conductivity.py:11](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/thermal_conductivity.py#L11)

Calculate the thermal conductivity of air as a function of temperature.
Based on a simplified linear relation from atmospheric science literature.
Only valid for temperatures within the range typically found on
Earth's surface.

#### Arguments

-----
- temperature (Union[float, NDArray[np.float64]]): The temperature at which
the thermal conductivity of air is to be calculated, in Kelvin (K).

#### Returns

--------
- Union[float, NDArray[np.float64]]: The thermal conductivity of air at the
specified temperature in Watts per meter-Kelvin (W/m·K) or J/(m s K).

#### Raises

------
- `-` *ValueError* - If the temperature is below absolute zero (0 K).

#### References

----------
- Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 17.54.

#### Signature

```python
def get_thermal_conductivity(
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
