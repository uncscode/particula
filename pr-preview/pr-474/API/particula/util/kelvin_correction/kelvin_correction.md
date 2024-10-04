# Kelvin Correction

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Kelvin Correction

> Auto-generated documentation for [particula.util.kelvin_correction](https://github.com/uncscode/particula/blob/main/particula/util/kelvin_correction.py) module.

## kelvin_radius

[Show source in kelvin_correction.py:11](https://github.com/uncscode/particula/blob/main/particula/util/kelvin_correction.py#L11)

 Kelvin radius (Neil's definition)
https://en.wikipedia.org/wiki/Kelvin_equation

#### Signature

```python
def kelvin_radius(
    surface_tension=0.072 * u.N / u.m,
    molecular_weight=0.01815 * u.kg / u.mol,
    density=1000 * u.kg / u.m**3,
    temperature=298.15 * u.K,
): ...
```



## kelvin_term

[Show source in kelvin_correction.py:31](https://github.com/uncscode/particula/blob/main/particula/util/kelvin_correction.py#L31)

 Kelvin term (Neil's definition)
https://en.wikipedia.org/wiki/Kelvin_equation

#### Signature

```python
def kelvin_term(radius=None, **kwargs): ...
```
