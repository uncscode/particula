# Concentration Function

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Concentration Function

> Auto-generated documentation for [particula.next.gas.properties.concentration_function](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/concentration_function.py) module.

## calculate_concentration

[Show source in concentration_function.py:11](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/concentration_function.py#L11)

Calculate the concentration of a gas from its partial pressure, molar mass,
and temperature using the ideal gas law.

#### Arguments

- pressure (float or NDArray[np.float64]): Partial pressure of the gas
in Pascals (Pa).
- molar_mass (float or NDArray[np.float64]): Molar mass of the gas in kg/mol
- temperature (float or NDArray[np.float64]): Temperature in Kelvin.

#### Returns

- concentration (float or NDArray[np.float64]): Concentration of the gas
in kg/m^3.

#### Signature

```python
def calculate_concentration(
    partial_pressure: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
