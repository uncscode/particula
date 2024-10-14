# Mean Free Path

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Mean Free Path

> Auto-generated documentation for [particula.next.gas.properties.mean_free_path](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/mean_free_path.py) module.

## molecule_mean_free_path

[Show source in mean_free_path.py:27](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/mean_free_path.py#L27)

Calculate the mean free path of a gas molecule in air based on the
temperature, pressure, and molar mass of the gas. The mean free path
is the average distance traveled by a molecule between collisions with
other molecules present in a medium (air).

#### Arguments

-----
- molar_mass (Union[float, NDArray[np.float64]]): The molar mass
of the gas molecule [kg/mol]. Default is the molecular weight of air.
- temperature (float): The temperature of the gas [K]. Default is 298.15 K.
- pressure (float): The pressure of the gas [Pa]. Default is 101325 Pa.
- dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
[Pa*s]. If not provided, it will be calculated based on the temperature.

#### Returns

--------
- Union[float, NDArray[np.float64]]: The mean free path of the gas molecule
in meters (m).

#### References

----------
- https://en.wikipedia.org/wiki/Mean_free_path

#### Signature

```python
def molecule_mean_free_path(
    molar_mass: ignore = MOLECULAR_WEIGHT_AIR.m,
    temperature: float = 298.15,
    pressure: float = 101325,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```
