# Vapor Pressure Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Vapor Pressure Module

> Auto-generated documentation for [particula.next.gas.properties.vapor_pressure_module](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/vapor_pressure_module.py) module.

## antoine_vapor_pressure

[Show source in vapor_pressure_module.py:14](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/vapor_pressure_module.py#L14)

Calculate vapor pressure using the Antoine equation.

#### Arguments

a, b, c: Antoine equation parameters.
- `temperature` - Temperature in Kelvin.

#### Returns

Vapor pressure in Pascals.

#### References

- `-` *Equation* - log10(P) = a - b / (T - c)
- https://en.wikipedia.org/wiki/Antoine_equation (but in Kelvin)
- Kelvin form:
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1

#### Signature

```python
def antoine_vapor_pressure(
    a: Union[float, NDArray[np.float64]],
    b: Union[float, NDArray[np.float64]],
    c: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## buck_vapor_pressure

[Show source in vapor_pressure_module.py:72](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/vapor_pressure_module.py#L72)

Calculate vapor pressure using the Buck equation for water vapor.

#### Arguments

- `temperature` - Temperature in Kelvin.

#### Returns

Vapor pressure in Pascals.

#### References

- Buck, A. L., 1981: New Equations for Computing Vapor Pressure and
    Enhancement Factor. J. Appl. Meteor. Climatol., 20, 1527-1532,
    https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2.
- https://en.wikipedia.org/wiki/Arden_Buck_equation

#### Signature

```python
def buck_vapor_pressure(
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## clausius_clapeyron_vapor_pressure

[Show source in vapor_pressure_module.py:42](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/vapor_pressure_module.py#L42)

Calculate vapor pressure using Clausius-Clapeyron equation.

#### Arguments

- `latent_heat` - Latent heat of vaporization in J/mol.
- `temperature_initial` - Initial temperature in Kelvin.
- `pressure_initial` - Initial vapor pressure in Pascals.
- `temperature` - Final temperature in Kelvin.
- `gas_constant` - gas constant (default is 8.314 J/(mol·K)).

#### Returns

Pure vapor pressure in Pascals.

#### References

- https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation

#### Signature

```python
def clausius_clapeyron_vapor_pressure(
    latent_heat: Union[float, NDArray[np.float64]],
    temperature_initial: Union[float, NDArray[np.float64]],
    pressure_initial: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    gas_constant: float = GAS_CONSTANT.m,
) -> Union[float, NDArray[np.float64]]: ...
```
