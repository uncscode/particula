# Knudsen Number

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Knudsen Number

> Auto-generated documentation for [particula.util.knudsen_number](https://github.com/uncscode/particula/blob/main/particula/util/knudsen_number.py) module.

## knu

[Show source in knudsen_number.py:18](https://github.com/uncscode/particula/blob/main/particula/util/knudsen_number.py#L18)

Returns particle's Knudsen number.

The Knudsen number reflects the relative length scales of
the particle and the suspending fluid (air, water, etc.).
This is calculated by the mean free path of the medium
divided by the particle radius.

The Knudsen number is a measure of continuum effects and
deviation thereof. For larger particles, the Knudsen number
goes towards 0. For smaller particles, the Knudsen number
goes towards infinity.

#### Examples

```
>>> from particula import u
>>> from particula.util.knudsen_number import knu
>>> # with radius 1e-9 m
>>> knu(radius=1e-9)
<Quantity(66.4798498, 'dimensionless')>
>>> # with radius 1e-9 m and mfp 60 nm
>>> knu(radius=1e-9*u.m, mfp=60*u.nm).m
60.00000000000001
>>> calculating via mfp(**kwargs)
>>> knu(
... radius=1e-9*u.m,
... temperature=300,
... pressure=1e5,
... molecular_weight=0.03,
... )
<Quantity(66.7097062, 'dimensionless')>
```

#### Arguments

radius  (float) [m]
mfp     (float) [m] (default: util)

#### Returns

(float) [dimensionless]

#### Notes

mfp can be calculated using mfp(**kwargs);
refer to particula.util.mean_free_path.mfp for more info.

#### Signature

```python
def knu(radius=None, mfp=None, **kwargs): ...
```
