# Dynamic Viscosity

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Gas](../index.md#gas) / [Properties](./index.md#properties) / Dynamic Viscosity

> Auto-generated documentation for [particula.next.gas.properties.dynamic_viscosity](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/dynamic_viscosity.py) module.

## get_dynamic_viscosity

[Show source in dynamic_viscosity.py:22](https://github.com/uncscode/particula/blob/main/particula/next/gas/properties/dynamic_viscosity.py#L22)

Calculates the dynamic viscosity of air via Sutherland's formula, which is
a common method in fluid dynamics for gases that involves temperature
adjustments.

#### Arguments

-----
- `-` *temperature* - Desired air temperature [K]. Must be greater than 0.
- `-` *reference_viscosity* - Gas viscosity [Pa*s] at the reference temperature
(default is STP).
- `-` *reference_temperature* - Gas temperature [K] for the reference viscosity
(default is STP).

#### Returns

--------
- `-` *float* - The dynamic viscosity of air at the given temperature [Pa*s].

#### Raises

------
- `-` *ValueError* - If the temperature is less than or equal to 0.

#### References

----------
https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula

#### Signature

```python
def get_dynamic_viscosity(
    temperature: float,
    reference_viscosity: float = REF_VISCOSITY_AIR_STP.m,
    reference_temperature: float = REF_TEMPERATURE_STP.m,
) -> float: ...
```
