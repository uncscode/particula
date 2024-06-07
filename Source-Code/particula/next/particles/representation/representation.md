# Representation

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Representation

> Auto-generated documentation for [particula.next.particles.representation](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py) module.

## ParticleRepresentation

[Show source in representation.py:14](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L14)

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

#### Signature

```python
class ParticleRepresentation:
    def __init__(
        self,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float_],
        density: NDArray[np.float_],
        concentration: NDArray[np.float_],
        charge: NDArray[np.float_],
    ): ...
```

#### See also

- [ActivityStrategy](./activity_strategies.md#activitystrategy)
- [DistributionStrategy](./distribution_strategies.md#distributionstrategy)
- [SurfaceStrategy](./surface_strategies.md#surfacestrategy)

### ParticleRepresentation().add_concentration

[Show source in representation.py:99](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L99)

Adds concentration to the particle distribution.

#### Arguments

- `-` *added_concentration* - The concentration to be
    added per distribution bin.

#### Signature

```python
def add_concentration(self, added_concentration: NDArray[np.float_]) -> None: ...
```

### ParticleRepresentation().add_mass

[Show source in representation.py:88](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L88)

Adds mass to the particle distribution, and updates parameters.

#### Arguments

- `-` *added_mass* - The mass to be added per
    distribution bin.

#### Signature

```python
def add_mass(self, added_mass: NDArray[np.float_]) -> None: ...
```

### ParticleRepresentation().get_charge

[Show source in representation.py:67](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L67)

Returns the charge per particle.

#### Returns

The charge of the particles.

#### Signature

```python
def get_charge(self) -> NDArray[np.float_]: ...
```

### ParticleRepresentation().get_mass

[Show source in representation.py:51](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L51)

Returns the mass of the particles as calculated by the strategy.

#### Returns

The mass of the particles.

#### Signature

```python
def get_mass(self) -> NDArray[np.float_]: ...
```

### ParticleRepresentation().get_radius

[Show source in representation.py:59](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L59)

Returns the radius of the particles as calculated by the strategy.

#### Returns

The radius of the particles.

#### Signature

```python
def get_radius(self) -> NDArray[np.float_]: ...
```

### ParticleRepresentation().get_total_mass

[Show source in representation.py:75](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/representation.py#L75)

Returns the total mass of the particles.

The total mass is as calculated by the strategy, taking into account
the distribution and concentration.

#### Returns

- `np.float_` - The total mass of the particles.

#### Signature

```python
def get_total_mass(self) -> np.float_: ...
```
