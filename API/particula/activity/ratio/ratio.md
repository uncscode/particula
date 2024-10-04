# Ratio

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Ratio

> Auto-generated documentation for [particula.activity.ratio](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py) module.

## from_molar_mass_ratio

[Show source in ratio.py:22](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py#L22)

Convert the given molar mass ratio (MW water / MW organic) to a
molar mass with respect to the other compound.

#### Arguments

- `molar_mass_ratio` *np.array* - The molar mass ratio with respect to water.
- `other_molar_mass` *float, optional* - The molar mass of the other compound.
    Defaults to 18.01528.

#### Returns

- `np.array` - The molar mass of the organic compound.

#### Signature

```python
def from_molar_mass_ratio(molar_mass_ratio, other_molar_mass=18.01528): ...
```



## to_molar_mass_ratio

[Show source in ratio.py:4](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py#L4)

Convert the given molar mass to a molar mass ratio with respect to water.
(MW water / MW organic)

#### Arguments

- `molar_mass` *np.array* - The molar mass of the organic compound.
- `other_molar_mass` *float, optional* - The molar mass of the other compound.
    Defaults to 18.01528.

#### Returns

- `np.array` - The molar mass ratio with respect to water.

#### Signature

```python
def to_molar_mass_ratio(molar_mass, other_molar_mass=18.01528): ...
```
