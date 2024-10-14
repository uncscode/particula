# Representation

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation

> Auto-generated documentation for [particula.next.particles.representation](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py) module.

## ParticleRepresentation

[Show source in representation.py:17](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L17)

Everything needed to represent a particle or a collection of particles.

Represents a particle or a collection of particles, encapsulating the
strategy for calculating mass, radius, and total mass based on a
specified particle distribution, density, and concentration. This class
allows for flexibility in representing particles.

#### Attributes

- `strategy` - The computation strategy for particle representations.
- `activity` - The activity strategy for the partial pressure calculations.
- `surface` - The surface strategy for surface tension and Kelvin effect.
- `distribution` - The distribution data for the particles, which could
    represent sizes, masses, or another relevant metric.
- `density` - The density of the material from which the particles are made.
- `concentration` - The concentration of particles within the distribution.
- `charge` - The charge on each particle.
- `volume` - The air volume for simulation of particles in the air,
    default is 1 m^3. This is only used in ParticleResolved Strategies.

#### Signature

```python
class ParticleRepresentation:
    def __init__(
        self,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float64],
        density: NDArray[np.float64],
        concentration: NDArray[np.float64],
        charge: NDArray[np.float64],
        volume: float = 1,
    ): ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)
- [DistributionStrategy](./distribution_strategies.md#distributionstrategy)
- [SurfaceStrategy](./surface_strategies.md#surfacestrategy)

### ParticleRepresentation().__str__

[Show source in representation.py:58](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L58)

Returns a string representation of the particle representation.

#### Returns

- `str` - A string representation of the particle representation.

#### Signature

```python
def __str__(self) -> str: ...
```

### ParticleRepresentation().add_concentration

[Show source in representation.py:309](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L309)

Adds concentration to the particle distribution.

#### Arguments

- `added_concentration` - The concentration to be
    added per distribution bin.

#### Signature

```python
def add_concentration(
    self,
    added_concentration: NDArray[np.float64],
    added_distribution: Optional[NDArray[np.float64]] = None,
) -> None: ...
```

### ParticleRepresentation().add_mass

[Show source in representation.py:294](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L294)

Adds mass to the particle distribution, and updates parameters.

#### Arguments

- `added_mass` - The mass to be added per
    distribution bin.

#### Signature

```python
def add_mass(self, added_mass: NDArray[np.float64]) -> None: ...
```

### ParticleRepresentation().collide_pairs

[Show source in representation.py:334](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L334)

Collide pairs of indices, used for ParticleResolved Strategies.

#### Arguments

- `indices` - The indices to collide.

#### Signature

```python
def collide_pairs(self, indices: NDArray[np.int64]) -> None: ...
```

### ParticleRepresentation().get_activity

[Show source in representation.py:96](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L96)

Returns the activity strategy used for partial pressure
calculations.

#### Arguments

- `clone` - If True, then return a deepcopy of the activity strategy.

#### Returns

The activity strategy used for partial pressure calculations.

#### Signature

```python
def get_activity(self, clone: bool = False) -> ActivityStrategy: ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)

### ParticleRepresentation().get_activity_name

[Show source in representation.py:110](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L110)

Returns the name of the activity strategy used for partial pressure
calculations.

#### Returns

The name of the activity strategy used for partial pressure
calculations.

#### Signature

```python
def get_activity_name(self) -> str: ...
```

### ParticleRepresentation().get_charge

[Show source in representation.py:198](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L198)

Returns the charge per particle.

#### Arguments

- `clone` - If True, then return a copy of the charge array.

#### Returns

The charge of the particles.

#### Signature

```python
def get_charge(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_concentration

[Show source in representation.py:170](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L170)

Returns the volume concentration of the particles.

For ParticleResolved Strategies, the concentration is the number of
particles per self.volume to get concentration/m^3. For other
Strategies, the concentration is the already per 1/m^3.

#### Arguments

- `clone` - If True, then return a copy of the concentration array.

#### Returns

The concentration of the particles.

#### Signature

```python
def get_concentration(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_density

[Show source in representation.py:157](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L157)

Returns the density of the particles.

#### Arguments

- `clone` - If True, then return a copy of the density array.

#### Returns

The density of the particles.

#### Signature

```python
def get_density(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_distribution

[Show source in representation.py:144](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L144)

Returns the distribution of the particles.

#### Arguments

- `clone` - If True, then return a copy of the distribution array.

#### Returns

The distribution of the particles.

#### Signature

```python
def get_distribution(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_mass

[Show source in representation.py:239](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L239)

Returns the mass of the particles as calculated by the strategy.

#### Arguments

- `clone` - If True, then return a copy of the mass array.

#### Returns

The mass of the particles.

#### Signature

```python
def get_mass(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_mass_concentration

[Show source in representation.py:254](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L254)

Returns the total mass / volume simulated.

The mass concentration is as calculated by the strategy, taking into
account the distribution and concentration.

#### Arguments

- `clone` - If True, then return a copy of the mass concentration.

#### Returns

- `np.float64` - The mass concentration of the particles, kg/m^3.

#### Signature

```python
def get_mass_concentration(self, clone: bool = False) -> np.float64: ...
```

### ParticleRepresentation().get_radius

[Show source in representation.py:280](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L280)

Returns the radius of the particles as calculated by the strategy.

#### Arguments

- `clone` - If True, then return a copy of the radius array

#### Returns

The radius of the particles.

#### Signature

```python
def get_radius(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_species_mass

[Show source in representation.py:224](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L224)

Returns the masses per species in the particles.

#### Arguments

- `clone` - If True, then return a copy of the mass array.

#### Returns

The mass of the particles per species.

#### Signature

```python
def get_species_mass(self, clone: bool = False) -> NDArray[np.float64]: ...
```

### ParticleRepresentation().get_strategy

[Show source in representation.py:75](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L75)

Returns the strategy used for particle representation.

#### Arguments

- `clone` - If True, then return a deepcopy of the strategy.

#### Returns

The strategy used for particle representation.

#### Signature

```python
def get_strategy(self, clone: bool = False) -> DistributionStrategy: ...
```

#### See also

- [DistributionStrategy](./distribution_strategies.md#distributionstrategy)

### ParticleRepresentation().get_strategy_name

[Show source in representation.py:88](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L88)

Returns the name of the strategy used for particle representation.

#### Returns

The name of the strategy used for particle representation.

#### Signature

```python
def get_strategy_name(self) -> str: ...
```

### ParticleRepresentation().get_surface

[Show source in representation.py:120](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L120)

Returns the surface strategy used for surface tension and
Kelvin effect.

#### Arguments

- `clone` - If True, then return a deepcopy of the surface strategy.

#### Returns

The surface strategy used for surface tension and Kelvin effect.

#### Signature

```python
def get_surface(self, clone: bool = False) -> SurfaceStrategy: ...
```

#### See also

- [SurfaceStrategy](./surface_strategies.md#surfacestrategy)

### ParticleRepresentation().get_surface_name

[Show source in representation.py:134](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L134)

Returns the name of the surface strategy used for surface tension
and Kelvin effect.

#### Returns

The name of the surface strategy used for surface tension and
Kelvin effect.

#### Signature

```python
def get_surface_name(self) -> str: ...
```

### ParticleRepresentation().get_total_concentration

[Show source in representation.py:187](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L187)

Returns the total concentration of the particles.

#### Arguments

- `clone` - If True, then return a copy of the concentration array.

#### Returns

The concentration of the particles.

#### Signature

```python
def get_total_concentration(self, clone: bool = False) -> np.float64: ...
```

### ParticleRepresentation().get_volume

[Show source in representation.py:211](https://github.com/uncscode/particula/blob/main/particula/next/particles/representation.py#L211)

Returns the volume of the particles.

#### Arguments

- `clone` - If True, then return a copy of the volume array.

#### Returns

The volume of the particles.

#### Signature

```python
def get_volume(self, clone: bool = False) -> float: ...
```
