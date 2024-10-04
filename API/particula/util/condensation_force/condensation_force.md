# Condensation Force

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Condensation Force

> Auto-generated documentation for [particula.util.condensation_force](https://github.com/uncscode/particula/blob/main/particula/util/condensation_force.py) module.

## condensation_force

[Show source in condensation_force.py:9](https://github.com/uncscode/particula/blob/main/particula/util/condensation_force.py#L9)

calculate the condensation driving force

Equation (9) in https://www.nature.com/articles/nature18271

#### Signature

```python
def condensation_force(
    vapor_concentraton, sat_vapor_concentration, particle_activity=None, **kwargs
): ...
```



## particle_activity_fun

[Show source in condensation_force.py:29](https://github.com/uncscode/particula/blob/main/particula/util/condensation_force.py#L29)

calculate the particle activity

Equation (9--10) in https://www.nature.com/articles/nature18271

#### Signature

```python
def particle_activity_fun(
    mass_fraction, activity_coefficient, kelvin_term=None, **kwargs
): ...
```
