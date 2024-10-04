# Rms Speed

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Rms Speed

> Auto-generated documentation for [particula.util.rms_speed](https://github.com/uncscode/particula/blob/main/particula/util/rms_speed.py) module.

## cbar

[Show source in rms_speed.py:10](https://github.com/uncscode/particula/blob/main/particula/util/rms_speed.py#L10)

Returns the mean speed of molecules in an ideal gas.

#### Arguments

temperature           (float) [K]      (default: 298.15)
molecular_weight      (float) [kg/mol] (default: constants)

#### Returns

(float) [m/s]

Using particula.constants:
    GAS_CONSTANT            (float) [J/mol/K]
    MOLECULAR_WEIGHT_AIR    (float) [kg/mol]

#### Signature

```python
def cbar(
    temperature=298.15, molecular_weight=MOLECULAR_WEIGHT_AIR, gas_constant=GAS_CONSTANT
): ...
```

#### See also

- [GAS_CONSTANT](../constants.md#gas_constant)
- [MOLECULAR_WEIGHT_AIR](../constants.md#molecular_weight_air)
