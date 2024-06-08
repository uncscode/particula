# Distribution Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Distribution Strategies

> Auto-generated documentation for [particula.next.particles.distribution_strategies](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py) module.

## DistributionStrategy

[Show source in distribution_strategies.py:11](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L11)

Abstract base class for particle strategy, defining the common
interface for mass, radius, and total mass calculations for different
particle representations.

#### Methods

- `get_mass` - Calculates the mass of particles.
- `get_radius` - Calculates the radius of particles.
- `get_total_mass` - Calculates the total mass of particles.
- `add_mass` - Adds mass to the distribution of particles.

#### Signature

```python
class DistributionStrategy(ABC): ...
```

### DistributionStrategy().add_mass

[Show source in distribution_strategies.py:84](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L84)

Adds mass to the distribution of particles based on their distribution,
concentration, and density.

#### Arguments

- distribution (NDArray[np.float_]): The distribution representation
of particles
- concentration (NDArray[np.float_]): The concentration of each
particle in the distribution.
- density (NDArray[np.float_]): The density of the particles.
- added_mass (NDArray[np.float_]): The mass to be added per
distribution bin.

#### Returns

- `-` *NDArray[np.float_]* - The new concentration array.
- `-` *NDArray[np.float_]* - The new distribution array.

#### Signature

```python
@abstractmethod
def add_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
    added_mass: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_]]: ...
```

### DistributionStrategy().get_mass

[Show source in distribution_strategies.py:24](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L24)

Calculates the mass of particles based on their distribution and
density.

#### Arguments

- distribution (NDArray[np.float_]): The distribution of particle
    sizes or masses.
- density (NDArray[np.float_]): The density of the particles.

#### Returns

- `-` *NDArray[np.float_]* - The mass of the particles.

#### Signature

```python
@abstractmethod
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### DistributionStrategy().get_radius

[Show source in distribution_strategies.py:43](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L43)

Calculates the radius of particles based on their distribution and
density.

#### Arguments

- distribution (NDArray[np.float_]): The distribution of particle
    sizes or masses.
- density (NDArray[np.float_]): The density of the particles.

#### Returns

- `-` *NDArray[np.float_]* - The radius of the particles.

#### Signature

```python
@abstractmethod
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### DistributionStrategy().get_total_mass

[Show source in distribution_strategies.py:62](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L62)

Calculates the total mass of particles based on their distribution,
concentration, and density.

#### Arguments

- distribution (NDArray[np.float_]): The distribution of particle
    sizes or masses.
- concentration (NDArray[np.float_]): The concentration of each
    particle size or mass in the distribution.
- density (NDArray[np.float_]): The density of the particles.

#### Returns

- `-` *np.float_* - The total mass of the particles.

#### Signature

```python
@abstractmethod
def get_total_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
) -> np.float_: ...
```



## MassBasedMovingBin

[Show source in distribution_strategies.py:111](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L111)

A strategy for particles represented by their mass distribution, and
particle number concentration. Moving the bins when adding mass.

#### Signature

```python
class MassBasedMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### MassBasedMovingBin().add_mass

[Show source in distribution_strategies.py:145](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L145)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
    added_mass: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_]]: ...
```

### MassBasedMovingBin().get_mass

[Show source in distribution_strategies.py:117](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L117)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### MassBasedMovingBin().get_radius

[Show source in distribution_strategies.py:125](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L125)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### MassBasedMovingBin().get_total_mass

[Show source in distribution_strategies.py:135](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L135)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
) -> np.float_: ...
```



## RadiiBasedMovingBin

[Show source in distribution_strategies.py:156](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L156)

A strategy for particles represented by their radius (distribution),
and particle concentration. Implementing the DistributionStrategy
interface.
This strategy calculates particle mass, radius, and total mass based on
the particle's radius, number concentration, and density.

#### Signature

```python
class RadiiBasedMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### RadiiBasedMovingBin().add_mass

[Show source in distribution_strategies.py:192](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L192)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
    added_mass: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_]]: ...
```

### RadiiBasedMovingBin().get_mass

[Show source in distribution_strategies.py:165](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L165)

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### RadiiBasedMovingBin().get_radius

[Show source in distribution_strategies.py:174](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L174)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### RadiiBasedMovingBin().get_total_mass

[Show source in distribution_strategies.py:181](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L181)

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
) -> np.float_: ...
```



## SpeciatedMassMovingBin

[Show source in distribution_strategies.py:204](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L204)

Strategy for particles with speciated mass distribution.
Some particles may have different densities and their mass is
distributed across different species. This strategy calculates mass,
radius, and total mass based on the species at each mass, density,
the particle concentration.

#### Signature

```python
class SpeciatedMassMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### SpeciatedMassMovingBin().add_mass

[Show source in distribution_strategies.py:280](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L280)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
    added_mass: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_]]: ...
```

### SpeciatedMassMovingBin().get_mass

[Show source in distribution_strategies.py:211](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L211)

Calculates the mass for each mass and species, leveraging densities
for adjustment.

#### Arguments

- distribution (NDArray[np.float_]): A 2D array with rows
    representing mass bins and columns representing species.
- densities (NDArray[np.float_]): An array of densities for each
    species.

#### Returns

- `-` *NDArray[np.float_]* - A 1D array of calculated masses for each mass
    bin. The sum of each column (species) in the distribution matrix.

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### SpeciatedMassMovingBin().get_radius

[Show source in distribution_strategies.py:235](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L235)

Calculates the radius for each mass bin and species, based on the
volume derived from mass and density.

#### Arguments

- distribution (NDArray[np.float_]): A 2D array with rows representing
    mass bins and columns representing species.
- density (NDArray[np.float_]): An array of densities for each
    species.

#### Returns

- `-` *NDArray[np.float_]* - A 1D array of calculated radii for each mass
    bin.

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float_], density: NDArray[np.float_]
) -> NDArray[np.float_]: ...
```

### SpeciatedMassMovingBin().get_total_mass

[Show source in distribution_strategies.py:258](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/distribution_strategies.py#L258)

Calculates the total mass of all species, incorporating the
concentration of particles per species.

#### Arguments

- distribution (NDArray[np.float_]): The mass distribution matrix.
- counts (NDArray[np.float_]): A 1D array with elements representing
    the count of particles for each species.

#### Returns

- `-` *np.float_* - The total mass of all particles.

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float_],
    concentration: NDArray[np.float_],
    density: NDArray[np.float_],
) -> np.float_: ...
```
