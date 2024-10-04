# Activity Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Activity Module

> Auto-generated documentation for [particula.next.particles.properties.activity_module](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/activity_module.py) module.

## calculate_partial_pressure

[Show source in activity_module.py:223](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/activity_module.py#L223)

Calculate the partial pressure of a species based on its activity and pure
    vapor pressure.

#### Arguments

pure_vapor_pressure (float or NDArray[np.float64]): Pure vapor pressure
    of the species in pascals (Pa).
activity (float or NDArray[np.float64]): Activity of the species,
    unitless.

#### Returns

float or NDArray[np.float64]: Partial pressure of the species in
pascals (Pa).

#### Examples

``` py title="Example"
calculate_partial_pressure(1000.0, 0.95)
# 950.0
```

#### Signature

```python
def calculate_partial_pressure(
    pure_vapor_pressure: Union[float, NDArray[np.float64]],
    activity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## ideal_activity_mass

[Show source in activity_module.py:88](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/activity_module.py#L88)

Calculate the ideal activity of a species based on mass fractions.

This function computes the activity based on the mass fractions of species
consistent with Raoult's Law.

#### Arguments

mass_concentration (float or NDArray[np.float64]): Mass concentration
    of the species in kilograms per cubic meter (kg/m^3).

#### Returns

float or NDArray[np.float64]]: Activity of the species, unitless.

#### Examples

``` py title="Example"
ideal_activity_mass(np.array([1.0, 2.0]))
array([0.3333, 0.6667])
```

#### References

- `Raoult's` *Law* - https://en.wikipedia.org/wiki/Raoult%27s_law

#### Signature

```python
def ideal_activity_mass(
    mass_concentration: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## ideal_activity_molar

[Show source in activity_module.py:10](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/activity_module.py#L10)

Calculate the ideal activity of a species based on mole fractions.

This function computes the activity based on the mole fractions of species
according to Raoult's Law.

#### Arguments

mass_concentration (float or NDArray[np.float64]): Mass concentration
    of the species in kilograms per cubic meter (kg/m^3).
molar_mass (float or NDArray[np.float64]): Molar mass of the species in
    kilograms per mole (kg/mol). A single value applies to all species
    if only one is provided.

#### Returns

float or NDArray[np.float64]: Activity of the species, unitless.

#### Examples

``` py title="Example"
ideal_activity_molar(
    mass_concentration=np.array([1.0, 2.0]),
    molar_mass=np.array([18.015, 28.97]))
# array([0.0525, 0.0691])
```

#### References

- `Raoult's` *Law* - https://en.wikipedia.org/wiki/Raoult%27s_law

#### Signature

```python
def ideal_activity_molar(
    mass_concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## ideal_activity_volume

[Show source in activity_module.py:49](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/activity_module.py#L49)

Calculate the ideal activity of a species based on volume fractions.

This function computes the activity based on the volume fractions of
species consistent with Raoult's Law.

#### Arguments

mass_concentration (float or NDArray[np.float64]): Mass concentration
    of the species in kilograms per cubic meter (kg/m^3).
density (float or NDArray[np.float64]): Density of the species in
    kilograms per cubic meter (kg/m^3). A single value applies to all
    species if only one is provided.

#### Returns

float or NDArray[np.float64]: Activity of the species, unitless.

#### Examples

``` py title="Example"
ideal_activity_volume(
    mass_concentration=np.array([1.0, 2.0]),
    density=np.array([1000.0, 1200.0]))
# array([0.001, 0.002])
```

#### References

- `Raoult's` *Law* - https://en.wikipedia.org/wiki/Raoult%27s_law

#### Signature

```python
def ideal_activity_volume(
    mass_concentration: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## kappa_activity

[Show source in activity_module.py:121](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/activity_module.py#L121)

Calculate the activity of species based on the kappa hygroscopic parameter.

This function computes the activity using the kappa parameter and the
species' mass concentrations, considering the volume fractions and water
content.

#### Arguments

mass_concentration (float or NDArray[np.float64]]): Mass concentration
    of the species in kilograms per cubic meter (kg/m^3).
- `kappa` *NDArray[np.float64]* - Kappa hygroscopic parameter, unitless.
- `density` *NDArray[np.float64]* - Density of the species in kilograms per
    cubic meter (kg/m^3).
- `molar_mass` *NDArray[np.float64]* - Molar mass of the species in
    kilograms per mole (kg/mol).
- `water_index` *int* - Index of water in the mass concentration array.

#### Returns

- `NDArray[np.float64]` - Activity of the species, unitless.

#### Examples

``` py title="Example"
kappa_activity(
    mass_concentration=np.array([[1.0, 2.0], [3.0, 4.0]]),
    kappa=np.array([0.0, 0.2]),
    density=np.array([1000.0, 1200.0]),
    molar_mass=np.array([18.015, 28.97]),
    water_index=0
)
# array([[0.95, 0.75], [0.85, 0.65]])
```

#### References

Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
representation of hygroscopic growth and cloud condensation nucleus
activity. Atmospheric Chemistry and Physics, 7(8), 1961-1971.
- `DOI` - https://doi.org/10.5194/acp-7-1961-2007

#### Signature

```python
def kappa_activity(
    mass_concentration: NDArray[np.float64],
    kappa: NDArray[np.float64],
    density: NDArray[np.float64],
    molar_mass: NDArray[np.float64],
    water_index: int,
) -> NDArray[np.float64]: ...
```
