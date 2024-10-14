# Surface Tension

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Surface Tension

> Auto-generated documentation for [particula.util.surface_tension](https://github.com/uncscode/particula/blob/main/particula/util/surface_tension.py) module.

## dry_mixing

[Show source in surface_tension.py:42](https://github.com/uncscode/particula/blob/main/particula/util/surface_tension.py#L42)

Function to calculate the effective surface tension of a dry mixture.

#### Arguments

-----
- volume_fractions : array, volume fractions of solutes
- surface_tensions : array, surface tensions of solutes

#### Returns

--------
- sigma : array, surface tension of droplet

#### Signature

```python
def dry_mixing(volume_fractions, surface_tensions): ...
```



## water

[Show source in surface_tension.py:15](https://github.com/uncscode/particula/blob/main/particula/util/surface_tension.py#L15)

Calculate the surface tension of water using the equation from Kalova
and Mares (2018).

#### Arguments

-----
- Temperature : float, Ambient temperature of air
- CritTemp : float, optional: Critical temperature of water

#### Returns

-------
- sigma : float, Surface tension of water at the given temperature

#### Signature

```python
def water(temperature, critical_temperature=647.15): ...
```



## wet_mixing

[Show source in surface_tension.py:64](https://github.com/uncscode/particula/blob/main/particula/util/surface_tension.py#L64)

Function to calculate the effective surface tension of a wet mixture.

#### Arguments

----------
- volume_solute : array, volume of solute mixture
- volume_water : array, volume of water
- surface_tension_solute : array, surface tension of solute mixture
- temperature : float, temperature of droplet
- method : str, optional: [film, volume] method to calculate effective
    surface tension

#### Returns

--------
- EffSigma : array, effective surface tension of droplet

#### Signature

```python
def wet_mixing(
    volume_solute,
    volume_water,
    wet_radius,
    surface_tension_solute,
    temperature,
    method="film",
): ...
```
