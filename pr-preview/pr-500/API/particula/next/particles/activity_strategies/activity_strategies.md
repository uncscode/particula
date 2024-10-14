# Activity Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Activity Strategies

> Auto-generated documentation for [particula.next.particles.activity_strategies](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py) module.

## ActivityIdealMass

[Show source in activity_strategies.py:115](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L115)

Calculate ideal activity based on mass fractions.

This strategy utilizes mass fractions to determine the activity, consistent
with the principles outlined in Raoult's Law.

#### References

Mass Based [Raoult's Law](https://en.wikipedia.org/wiki/Raoult%27s_law)

#### Signature

```python
class ActivityIdealMass(ActivityStrategy): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### ActivityIdealMass().activity

[Show source in activity_strategies.py:125](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L125)

Calculate the activity of a species based on mass concentration.

#### Arguments

- `mass_concentration` - Concentration of the species in kilograms
per cubic meter (kg/m^3).

#### Returns

- `Union[float,` *NDArray[np.float64]]* - Activity of the particle,
unitless.

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ActivityIdealMolar

[Show source in activity_strategies.py:79](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L79)

Calculate ideal activity based on mole fractions.

This strategy uses mole fractions to compute the activity, adhering to
the principles of Raoult's Law.

#### Arguments

molar_mass (Union[float, NDArray[np.float64]]): Molar mass of the
species [kg/mol]. A single value applies to all species if only one
is provided.

#### References

Molar [Raoult's Law](https://en.wikipedia.org/wiki/Raoult%27s_law)

#### Signature

```python
class ActivityIdealMolar(ActivityStrategy):
    def __init__(self, molar_mass: Union[float, NDArray[np.float64]] = 0.0): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### ActivityIdealMolar().activity

[Show source in activity_strategies.py:97](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L97)

Calculate the activity of a species based on mass concentration.

#### Arguments

- `mass_concentration` - Concentration of the species in kilograms per
cubic meter (kg/m^3).

#### Returns

- `Union[float,` *NDArray[np.float64]]* - Activity of the species,
unitless.

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ActivityIdealVolume

[Show source in activity_strategies.py:141](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L141)

Calculate ideal activity based on volume fractions.

This strategy uses volume fractions to compute the activity, following
the principles of Raoult's Law.

#### References

Volume Based
    [Raoult's Law](https://en.wikipedia.org/wiki/Raoult%27s_law)

#### Signature

```python
class ActivityIdealVolume(ActivityStrategy):
    def __init__(self, density: Union[float, NDArray[np.float64]] = 0.0): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### ActivityIdealVolume().activity

[Show source in activity_strategies.py:155](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L155)

Calculate the activity of a species based on mass concentration.

#### Arguments

- `mass_concentration` - Concentration of the species in kilograms per
    cubic meter (kg/m^3).
- `density` - Density of the species in kilograms per cubic meter
    (kg/m^3).

#### Returns

- `Union[float,` *NDArray[np.float64]]* - Activity of the particle,
unitless.

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ActivityKappaParameter

[Show source in activity_strategies.py:176](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L176)

Non-ideal activity strategy based on the kappa hygroscopic parameter.

This strategy calculates the activity using the kappa hygroscopic
parameter, a measure of hygroscopicity. The activity is determined by the
species' mass concentration along with the hygroscopic parameter.

#### Arguments

- `kappa` - Kappa hygroscopic parameter, unitless.
    Includes a value for water which is excluded in calculations.
- `density` - Density of the species in kilograms per
    cubic meter (kg/m^3).
- `molar_mass` - Molar mass of the species in kilograms
    per mole (kg/mol).
- `water_index` - Index of water in the mass concentration array.

#### Signature

```python
class ActivityKappaParameter(ActivityStrategy):
    def __init__(
        self,
        kappa: NDArray[np.float64] = np.array([0.0], dtype=np.float64),
        density: NDArray[np.float64] = np.array([0.0], dtype=np.float64),
        molar_mass: NDArray[np.float64] = np.array([0.0], dtype=np.float64),
        water_index: int = 0,
    ): ...
```

#### See also

- [ActivityStrategy](#activitystrategy)

### ActivityKappaParameter().activity

[Show source in activity_strategies.py:205](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L205)

Calculate the activity of a species based on mass concentration.

#### Arguments

- `mass_concentration` - Concentration of the species in kilograms per
cubic meter (kg/m^3).

#### Returns

- `Union[float,` *NDArray[np.float64]]* - Activity of the particle,
unitless.

#### References

Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
representation of hygroscopic growth and cloud condensation nucleus
activity. Atmospheric Chemistry and Physics, 7(8), 1961-1971.
[DOI](https://doi.org/10.5194/acp-7-1961-2007), see EQ 2 and 7.

#### Signature

```python
def activity(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```



## ActivityStrategy

[Show source in activity_strategies.py:22](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L22)

Abstract base class for vapor pressure strategies.

This interface is used for implementing strategies based on particle
activity calculations, specifically for calculating vapor pressures.

#### Methods

- `get_name` - Return the type of the activity strategy.
- `activity` - Calculate the activity of a species.
- `partial_pressure` - Calculate the partial pressure of a species in
    the mixture.

#### Signature

```python
class ActivityStrategy(ABC): ...
```

### ActivityStrategy().activity

[Show source in activity_strategies.py:35](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L35)

Calculate the activity of a species based on its mass concentration.

#### Arguments

- `mass_concentration` - Concentration of the species [kg/m^3]

#### Returns

float or NDArray[float]: Activity of the particle, unitless.

#### Signature

```python
@abstractmethod
def activity(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]: ...
```

### ActivityStrategy().get_name

[Show source in activity_strategies.py:48](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L48)

Return the type of the activity strategy.

#### Signature

```python
def get_name(self) -> str: ...
```

### ActivityStrategy().partial_pressure

[Show source in activity_strategies.py:52](https://github.com/uncscode/particula/blob/main/particula/next/particles/activity_strategies.py#L52)

Calculate the vapor pressure of species in the particle phase.

This method computes the vapor pressure based on the species' activity
considering its pure vapor pressure and mass concentration.

#### Arguments

- `pure_vapor_pressure` - Pure vapor pressure of the species in
pascals (Pa).
- `mass_concentration` - Concentration of the species in kilograms per
cubic meter (kg/m^3).

#### Returns

- `Union[float,` *NDArray[np.float64]]* - Vapor pressure of the particle
in pascals (Pa).

#### Signature

```python
def partial_pressure(
    self,
    pure_vapor_pressure: Union[float, NDArray[np.float64]],
    mass_concentration: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
