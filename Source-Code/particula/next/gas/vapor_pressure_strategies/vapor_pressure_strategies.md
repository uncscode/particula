# Vapor Pressure Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Vapor Pressure Strategies

> Auto-generated documentation for [particula.next.gas.vapor_pressure_strategies](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py) module.

## AntoineVaporPressureStrategy

[Show source in vapor_pressure_strategies.py:201](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L201)

Concrete implementation of the VaporPressureStrategy using the
Antoine equation for vapor pressure calculations.

#### Signature

```python
class AntoineVaporPressureStrategy(VaporPressureStrategy):
    def __init__(
        self,
        a: Union[float, NDArray[np.float_]] = 0.0,
        b: Union[float, NDArray[np.float_]] = 0.0,
        c: Union[float, NDArray[np.float_]] = 0.0,
    ): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### AntoineVaporPressureStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:216](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L216)

Calculate the pure (saturation) vapor pressure using the Antoine
equation.

#### Arguments

----
- temperature (float or NDArray[np.float_]): Temperature in Kelvin.

#### Returns

-------
- vapor_pressure (float or NDArray[np.float_]): The vapor pressure in
Pascals.

#### References

----------
- `-` *Equation* - log10(P) = a - b / (T - c)
- https://en.wikipedia.org/wiki/Antoine_equation (but in Kelvin)
- Kelvin form:
https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```



## ClausiusClapeyronStrategy

[Show source in vapor_pressure_strategies.py:247](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L247)

Concrete implementation of the VaporPressureStrategy using the
Clausius-Clapeyron equation for vapor pressure calculations.

#### Signature

```python
class ClausiusClapeyronStrategy(VaporPressureStrategy):
    def __init__(
        self,
        latent_heat: Union[float, NDArray[np.float_]],
        temperature_initial: Union[float, NDArray[np.float_]],
        pressure_initial: Union[float, NDArray[np.float_]],
    ): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### ClausiusClapeyronStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:274](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L274)

Calculate the vapor pressure at a new temperature using the
Clausius-Clapeyron equation. For ideal gases at low temperatures.

#### Arguments

----
- temperature_initial (float or NDArray[np.float_]): Initial
temperature in Kelvin.
- pressure_initial (float or NDArray[np.float_]): Initial vapor
pressure in Pascals.
- temperature_final (float or NDArray[np.float_]): Final temperature
in Kelvin.

#### Returns

- vapor_pressure_final (float or NDArray[np.float_]): Final vapor
pressure in Pascals.

#### References

----------
- https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
Ideal_gas_approximation_at_low_temperatures

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```



## ConstantVaporPressureStrategy

[Show source in vapor_pressure_strategies.py:175](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L175)

Concrete implementation of the VaporPressureStrategy using a constant
vapor pressure value.

#### Signature

```python
class ConstantVaporPressureStrategy(VaporPressureStrategy):
    def __init__(self, vapor_pressure: Union[float, NDArray[np.float_]]): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### ConstantVaporPressureStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:182](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L182)

Return the constant vapor pressure value.

#### Arguments

----
- temperature (float or NDArray[np.float_]): Not used.

#### Returns

-------
- vapor_pressure (float or NDArray[np.float_]): The constant vapor
pressure value in Pascals.

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```



## VaporPressureStrategy

[Show source in vapor_pressure_strategies.py:63](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L63)

Abstract class for vapor pressure calculations. The methods
defined here must be implemented by subclasses below.

#### Signature

```python
class VaporPressureStrategy(ABC): ...
```

### VaporPressureStrategy().concentration

[Show source in vapor_pressure_strategies.py:92](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L92)

Calculate the concentration of the gas at a given pressure and
temperature.

#### Arguments

----
- partial_pressure (float or NDArray[np.float_]): Pressure in Pascals.
- molar_mass (float or NDArray[np.float_]): Molar mass of the gas in
kg/mol.
- temperature (float or NDArray[np.float_]): Temperature in Kelvin.

#### Returns

-------
- concentration (float or NDArray[np.float_]): The concentration of the
gas in kg/m^3.

#### Signature

```python
def concentration(
    self,
    partial_pressure: Union[float, NDArray[np.float_]],
    molar_mass: Union[float, NDArray[np.float_]],
    temperature: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]: ...
```

### VaporPressureStrategy().partial_pressure

[Show source in vapor_pressure_strategies.py:67](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L67)

Calculate the partial pressure of the gas from its concentration, molar
mass, and temperature.

#### Arguments

----
- concentration (float or NDArray[np.float_]): Concentration of the gas
in kg/m^3.
- molar_mass (float or NDArray[np.float_]): Molar mass of the gas in
kg/mol.
- temperature (float or NDArray[np.float_]): Temperature in Kelvin.

#### Returns

-------
- partial_pressure (float or NDArray[np.float_]): Partial pressure of
the gas in Pascals.

#### Signature

```python
def partial_pressure(
    self,
    concentration: Union[float, NDArray[np.float_]],
    molar_mass: Union[float, NDArray[np.float_]],
    temperature: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]: ...
```

### VaporPressureStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:163](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L163)

Calculate the pure (saturation) vapor pressure at a given
temperature. Units are in Pascals Pa=kg/(m·s²).

#### Arguments

temperature (float or NDArray[np.float_]): Temperature in Kelvin.

#### Signature

```python
@abstractmethod
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```

### VaporPressureStrategy().saturation_concentration

[Show source in vapor_pressure_strategies.py:139](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L139)

Calculate the saturation concentration of the gas at a given
temperature.

#### Arguments

----
- molar_mass (float or NDArray[np.float_]): Molar mass of the gas in
kg/mol.
- temperature (float or NDArray[np.float_]): Temperature in Kelvin.

#### Returns

-------
- saturation_concentration (float or NDArray[np.float_]):
The saturation concentration of the gas in kg/m^3.

#### Signature

```python
def saturation_concentration(
    self,
    molar_mass: Union[float, NDArray[np.float_]],
    temperature: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]: ...
```

### VaporPressureStrategy().saturation_ratio

[Show source in vapor_pressure_strategies.py:116](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L116)

Calculate the saturation ratio of the gas at a given pressure and
temperature.

#### Arguments

----
- pressure (float or NDArray[np.float_]): Pressure in Pascals.
- temperature (float or NDArray[np.float_]): Temperature in Kelvin.

#### Returns

-------
- saturation_ratio (float or NDArray[np.float_]): The saturation ratio
of the gas.

#### Signature

```python
def saturation_ratio(
    self,
    concentration: Union[float, NDArray[np.float_]],
    molar_mass: Union[float, NDArray[np.float_]],
    temperature: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]: ...
```



## WaterBuckStrategy

[Show source in vapor_pressure_strategies.py:304](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L304)

Concrete implementation of the VaporPressureStrategy using the
Buck equation for water vapor pressure calculations.

#### Signature

```python
class WaterBuckStrategy(VaporPressureStrategy): ...
```

#### See also

- [VaporPressureStrategy](#vaporpressurestrategy)

### WaterBuckStrategy().pure_vapor_pressure

[Show source in vapor_pressure_strategies.py:308](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L308)

Calculate the pure (saturation) vapor pressure using the Buck
equation for water vapor.

#### Arguments

----
- temperature (float or NDArray[np.float_]): Temperature in Kelvin.

#### Returns

-------
- vapor_pressure (float or NDArray[np.float_]): The vapor pressure in
Pascals.

#### References

----------
Buck, A. L., 1981: New Equations for Computing Vapor Pressure and
Enhancement Factor. J. Appl. Meteor. Climatol., 20, 1527-1532,
https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2.

https://en.wikipedia.org/wiki/Arden_Buck_equation

#### Signature

```python
def pure_vapor_pressure(
    self, temperature: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]: ...
```



## calculate_concentration

[Show source in vapor_pressure_strategies.py:40](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L40)

Calculate the concentration of a gas from its partial pressure, molar mass,
and temperature using the ideal gas law.

#### Arguments

- pressure (float or NDArray[np.float_]): Partial pressure of the gas
in Pascals (Pa).
- molar_mass (float or NDArray[np.float_]): Molar mass of the gas in kg/mol
- temperature (float or NDArray[np.float_]): Temperature in Kelvin.

#### Returns

- concentration (float or NDArray[np.float_]): Concentration of the gas
in kg/m^3.

#### Signature

```python
def calculate_concentration(
    partial_pressure: Union[float, NDArray[np.float_]],
    molar_mass: Union[float, NDArray[np.float_]],
    temperature: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]: ...
```



## calculate_partial_pressure

[Show source in vapor_pressure_strategies.py:18](https://github.com/Gorkowski/particula/blob/main/particula/next/gas/vapor_pressure_strategies.py#L18)

Calculate the partial pressure of a gas from its concentration, molar mass,
and temperature.

#### Arguments

- concentration (float): Concentration of the gas in kg/m^3.
- molar_mass (float): Molar mass of the gas in kg/mol.
- temperature (float): Temperature in Kelvin.

#### Returns

- `-` *float* - Partial pressure of the gas in Pascals (Pa).

#### Signature

```python
def calculate_partial_pressure(
    concentration: Union[float, NDArray[np.float_]],
    molar_mass: Union[float, NDArray[np.float_]],
    temperature: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]: ...
```
