# Friction Factor

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Friction Factor

> Auto-generated documentation for [particula.util.friction_factor](https://github.com/uncscode/particula/blob/main/particula/util/friction_factor.py) module.

## frifac

[Show source in friction_factor.py:10](https://github.com/uncscode/particula/blob/main/particula/util/friction_factor.py#L10)

Returns a particle's friction factor.

Property of the particle's size and surrounding medium.
Multiplying the friction factor by the fluid velocity
yields the drag force on the particle.

It is best thought of as an inverse of mobility or
the ratio between thermal energy and diffusion coefficient.
The modified Stoke's diffusion coefficient is defined as
kT / (6 * np.pi * dyn_vis_air * radius / slip_corr)
and thus the friction factor can be defined as
(6 * np.pi * dyn_vis_air * radius / slip_corr).

In the continuum limit (Kn -> 0; Cc -> 1):
    6 * np.pi * dyn_vis_air * radius

In the kinetic limit (Kn -> inf):
    8.39 * (dyn_vis_air/mfp_air) * const * radius**2

See more: DOI: 10.1080/02786826.2012.690543 (const=1.36)

#### Examples

```
>>> from particula import u
>>> from particula.util.friction_factor import frifac
>>> # with 1e-9 m radius
>>> frifac(radius=1e-9)
<Quantity(3.12763919e-15, 'kilogram / second')>
>>> # with 1e-9 m radius and 1e-5 N/m^2 dynamic viscosity
>>> frifac(radius=1e-9, dynamic_viscosity=1e-5)
<Quantity(3.114213e-15, 'kilogram / second')>
>>> # using dyn_vis(**kwargs)
>>> frifac(
... radius=1e-9,
... temperature=298.15,
... reference_viscosity=1.716e-5,
... reference_temperature=273.15
)
<Quantity(3.12763919e-15, 'kilogram / second')>
>>> # overriding sfc(**kwargs)
>>> frifac(radius=1e-9, slip_correction=1.5)
<Quantity(3.12763919e-15, 'kilogram / second')>
```

#### Arguments

radius            (float) [m]
dynamic_viscosity (float) [kg/m/s]  (default: util)
slip_corr_factor  (float) [ ]       (default: util)

#### Returns

(float) [N*s/m]

#### Notes

dynamic_viscosity can be calculated using the utility
function particula.util.dynamic_viscosity.dyn_vis(**kwargs)
and slip_corr_factor can be calculated using the utility
function particula.util.slip_correction.scf(**kwargs);
see respective documentation for more information.

#### Signature

```python
def frifac(radius=None, dynamic_viscosity=None, scf_val=None, **kwargs): ...
```
