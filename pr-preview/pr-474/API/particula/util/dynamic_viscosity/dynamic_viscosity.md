# Dynamic Viscosity

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Dynamic Viscosity

> Auto-generated documentation for [particula.util.dynamic_viscosity](https://github.com/uncscode/particula/blob/main/particula/util/dynamic_viscosity.py) module.

## dyn_vis

[Show source in dynamic_viscosity.py:21](https://github.com/uncscode/particula/blob/main/particula/util/dynamic_viscosity.py#L21)

 The dynamic viscosity of air via Sutherland formula.
This formula depends on temperature (temp) and the reference
temperature (t_ref) as well as the reference viscosity (mu_ref).

#### Examples

```
>>> from particula import u
>>> from particula.util.dynamic_viscosity import dyn_vis
>>> # with units
>>> dyn_vis(
... temperature=298.15*u.K,
... reference_viscosity=1.716e-5*u.Pa*u.s
... )
<Quantity(1.83714937e-05, 'kilogram / meter / second')>
>>> # without units and taking magnitude
>>> dyn_vis(
... temperature=298.15,
... reference_viscosity=1.716e-5
... ).magnitude
1.8371493734583912e-05
>>> # without units, all keyword arguments
>>> dyn_vis(
... temperature=298.15,
... reference_viscosity=1.716e-5,
... reference_temperature=273.15
... )
<Quantity(1.83714937e-05, 'kilogram / meter / second')>
>>> # for a list of temperatures
>>> dyn_vis(temperature=[200, 250, 300, 400]).m
array([1.32849751e-05, 1.59905239e-05, 1.84591625e-05, 2.28516090e-05])
```

Inputs:
    temperature             (float) [K]     (default: 298.15)
    reference_viscosity     (float) [Pa*s]  (default: constants)
    reference_temperature   (float) [K]     (default: constants)
    sutherland_constant     (float) [K]     (default: constants)

#### Returns

(float) [Pa*s]

Using particula.constants:
    REF_VISCOSITY_AIR_STP   (float) [Pa*s]
    REF_TEMPERATURE_STP     (float) [K]
    SUTHERLAND_CONSTANT     (float) [K]

#### Signature

```python
def dyn_vis(
    temperature=298.15 * u.K,
    reference_viscosity=REF_VISCOSITY_AIR_STP,
    reference_temperature=REF_TEMPERATURE_STP,
    sutherland_constant=SUTHERLAND_CONSTANT,
    **kwargs
): ...
```

#### See also

- [REF_TEMPERATURE_STP](../constants.md#ref_temperature_stp)
- [REF_VISCOSITY_AIR_STP](../constants.md#ref_viscosity_air_stp)
- [SUTHERLAND_CONSTANT](../constants.md#sutherland_constant)
