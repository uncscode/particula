# Surface Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Surface Strategies

> Auto-generated documentation for [particula.next.particles.surface_strategies](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py) module.

## SurfaceStrategy

[Show source in surface_strategies.py:18](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L18)

ABC class for Surface Strategies.

Abstract class for implementing strategies to calculate surface tension
and the Kelvin effect for species in particulate phases.

#### Methods

- `effective_surface_tension` - Calculate the effective surface tension of
    species based on their concentration.
- `effective_density` - Calculate the effective density of species based on
    their concentration.
- `get_name` - Return the type of the surface strategy.
- `kelvin_radius` - Calculate the Kelvin radius which determines the
    curvature effect on vapor pressure.
- `kelvin_term` - Calculate the Kelvin term, which quantifies the effect of
    particle curvature on vapor pressure.

#### Signature

```python
class SurfaceStrategy(ABC): ...
```

### SurfaceStrategy().effective_density

[Show source in surface_strategies.py:49](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L49)

Calculate the effective density of the species mixture.

#### Arguments

- `mass_concentration` - Concentration of the species [kg/m^3].

#### Returns

float or NDArray[float]: Effective density of the species [kg/m^3].

#### Signature

```python
@abstractmethod
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```

### SurfaceStrategy().effective_surface_tension

[Show source in surface_strategies.py:36](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L36)

Calculate the effective surface tension of the species mixture.

#### Arguments

- `mass_concentration` - Concentration of the species [kg/m^3].

#### Returns

float or NDArray[float]: Effective surface tension [N/m].

#### Signature

```python
@abstractmethod
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```

### SurfaceStrategy().get_name

[Show source in surface_strategies.py:62](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L62)

Return the type of the surface strategy.

#### Signature

```python
def get_name(self) -> str: ...
```

### SurfaceStrategy().kelvin_radius

[Show source in surface_strategies.py:66](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L66)

Calculate the Kelvin radius which determines the curvature effect.

The kelvin radius is molecule specific and depends on the surface
tension, molar mass, density, and temperature of the system. It is
used to calculate the Kelvin term, which quantifies the effect of
particle curvature on vapor pressure.

#### Arguments

- `surface_tension` - Surface tension of the mixture [N/m].
- `molar_mass` - Molar mass of the species [kg/mol].
- `mass_concentration` - Concentration of the species [kg/m^3].
- `temperature` - Temperature of the system [K].

#### Returns

float or NDArray[float]: Kelvin radius [m].

#### References

- Based on Neil Donahue's approach to the Kelvin equation:
r = 2 * surface_tension * molar_mass / (R * T * density)
[Kelvin Wikipedia](https://en.wikipedia.org/wiki/Kelvin_equation)

#### Signature

```python
def kelvin_radius(
    self,
    molar_mass: Union[float, NDArray[np.float64]],
    mass_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
) -> Union[float, NDArray[np.float64]]: ...
```

### SurfaceStrategy().kelvin_term

[Show source in surface_strategies.py:100](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L100)

Calculate the Kelvin term, which multiplies the vapor pressure.

The Kelvin term is used to adjust the vapor pressure of a species due
to the curvature of the particle.

#### Arguments

- `radius` - Radius of the particle [m].
- `molar_mass` - Molar mass of the species a [kg/mol].
- `mass_concentration` - Concentration of the species [kg/m^3].
- `temperature` - Temperature of the system [K].

#### Returns

float or NDArray[float]: The exponential factor adjusting vapor
    pressure due to curvature.

#### References

Based on Neil Donahue's approach to the Kelvin equation:
exp(kelvin_radius / particle_radius)
[Kelvin Eq Wikipedia](https://en.wikipedia.org/wiki/Kelvin_equation)

#### Signature

```python
def kelvin_term(
    self,
    radius: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    mass_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
) -> Union[float, NDArray[np.float64]]: ...
```



## SurfaceStrategyMass

[Show source in surface_strategies.py:186](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L186)

Surface tension and density, based on mass fraction weighted values.

#### Arguments

- `surface_tension` - Surface tension of the species [N/m]. If a single
    value is provided, it will be used for all species.
- `density` - Density of the species [kg/m^3]. If a single value is
    provided, it will be used for all species.

#### References

[Mass Fractions](https://en.wikipedia.org/wiki/Mass_fraction_(chemistry))

#### Signature

```python
class SurfaceStrategyMass(SurfaceStrategy):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
    ): ...
```

#### See also

- [SurfaceStrategy](#surfacestrategy)

### SurfaceStrategyMass().effective_density

[Show source in surface_strategies.py:219](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L219)

#### Signature

```python
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```

### SurfaceStrategyMass().effective_surface_tension

[Show source in surface_strategies.py:207](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L207)

#### Signature

```python
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```



## SurfaceStrategyMolar

[Show source in surface_strategies.py:134](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L134)

Surface tension and density, based on mole fraction weighted values.

#### Arguments

- `surface_tension` - Surface tension of the species [N/m]. If a single
    value is provided, it will be used for all species.
- `density` - Density of the species [kg/m^3]. If a single value is
    provided, it will be used for all species.
- `molar_mass` - Molar mass of the species [kg/mol]. If a single value is
    provided, it will be used for all species.

#### References

[Mole Fractions](https://en.wikipedia.org/wiki/Mole_fraction)

#### Signature

```python
class SurfaceStrategyMolar(SurfaceStrategy):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
        molar_mass: Union[float, NDArray[np.float64]] = 0.01815,
    ): ...
```

#### See also

- [SurfaceStrategy](#surfacestrategy)

### SurfaceStrategyMolar().effective_density

[Show source in surface_strategies.py:172](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L172)

#### Signature

```python
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```

### SurfaceStrategyMolar().effective_surface_tension

[Show source in surface_strategies.py:159](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L159)

#### Signature

```python
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```



## SurfaceStrategyVolume

[Show source in surface_strategies.py:230](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L230)

Surface tension and density, based on volume fraction weighted values.

#### Arguments

- `surface_tension` - Surface tension of the species [N/m]. If a single
    value is provided, it will be used for all species.
- `density` - Density of the species [kg/m^3]. If a single value is
    provided, it will be used for all species.

#### References

[Volume Fractions](https://en.wikipedia.org/wiki/Volume_fraction)

#### Signature

```python
class SurfaceStrategyVolume(SurfaceStrategy):
    def __init__(
        self,
        surface_tension: Union[float, NDArray[np.float64]] = 0.072,
        density: Union[float, NDArray[np.float64]] = 1000,
    ): ...
```

#### See also

- [SurfaceStrategy](#surfacestrategy)

### SurfaceStrategyVolume().effective_density

[Show source in surface_strategies.py:264](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L264)

#### Signature

```python
def effective_density(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```

### SurfaceStrategyVolume().effective_surface_tension

[Show source in surface_strategies.py:251](https://github.com/uncscode/particula/blob/main/particula/next/particles/surface_strategies.py#L251)

#### Signature

```python
def effective_surface_tension(
    self, mass_concentration: Union[float, NDArray[np.float64]]
) -> float: ...
```
