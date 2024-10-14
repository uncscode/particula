# Partitioning

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Equilibria](./index.md#equilibria) / Partitioning

> Auto-generated documentation for [particula.equilibria.partitioning](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py) module.

## get_properties_for_liquid_vapor_partitioning

[Show source in partitioning.py:212](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L212)

Get properties for liquid-vapor partitioning.

#### Signature

```python
def get_properties_for_liquid_vapor_partitioning(
    water_activity_desired, molar_mass, oxygen2carbon, density
): ...
```



## liquid_vapor_obj_function

[Show source in partitioning.py:9](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L9)

Objective function for liquid-vapor partitioning.

#### Signature

```python
def liquid_vapor_obj_function(
    e_j_partition_guess,
    c_star_j_dry,
    concentration_organic_matter,
    gamma_organic_ab,
    mass_fraction_water_ab,
    q_ab,
    molar_mass,
    error_only=True,
): ...
```



## liquid_vapor_partitioning

[Show source in partitioning.py:158](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L158)

Thermodynamic equilibrium between liquid and vapor phase.
with activity coefficients,

#### Signature

```python
def liquid_vapor_partitioning(
    c_star_j_dry,
    concentration_organic_matter,
    molar_mass,
    gamma_organic_ab,
    mass_fraction_water_ab,
    q_ab,
    partition_coefficient_guess=None,
): ...
```
