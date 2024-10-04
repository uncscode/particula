# Distribution Strategies

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Particles](./index.md#particles) / Distribution Strategies

> Auto-generated documentation for [particula.next.particles.distribution_strategies](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py) module.

## DistributionStrategy

[Show source in distribution_strategies.py:13](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L13)

Abstract base class for particle strategy, defining the common
interface for mass, radius, and total mass calculations for different
particle representations.

#### Methods

- `get_name` - Returns the type of the distribution strategy.
- `get_mass` - Calculates the mass of particles.
- `get_radius` - Calculates the radius of particles.
- `get_total_mass` - Calculates the total mass of particles.
- `add_mass` - Adds mass to the distribution of particles.

#### Signature

```python
class DistributionStrategy(ABC): ...
```

### DistributionStrategy().add_concentration

[Show source in distribution_strategies.py:120](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L120)

Adds concentration to the distribution of particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `concentration` - The concentration of each particle size or mass in
    the distribution.
- `added_distribution` - The distribution to be added.
- `added_concentration` - The concentration to be added.

#### Returns

- `(distribution,` *concentration)* - The new distribution array and the
    new concentration array.

#### Signature

```python
@abstractmethod
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### DistributionStrategy().add_mass

[Show source in distribution_strategies.py:98](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L98)

Adds mass to the distribution of particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `concentration` - The concentration of each particle size or mass in
    the distribution.
- `density` - The density of the particles.
- `added_mass` - The mass to be added per distribution bin.

#### Returns

- `(distribution,` *concentration)* - The new distribution array and the
    new concentration array.

#### Signature

```python
@abstractmethod
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### DistributionStrategy().collide_pairs

[Show source in distribution_strategies.py:142](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L142)

Collides index pairs.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `concentration` - The concentration of each particle size or mass in
    the distribution.
- `density` - The density of the particles.
- `indices` - The indices of the particles to collide.

#### Returns

- `(distribution,` *concentration)* - The new distribution array and the
    new concentration array.

#### Signature

```python
@abstractmethod
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### DistributionStrategy().get_mass

[Show source in distribution_strategies.py:45](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L45)

Calculates the mass of the particles (or bin).

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `density` - The density of the particles.

#### Returns

- `NDArray[np.float64]` - The mass of the particles.

#### Signature

```python
def get_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### DistributionStrategy().get_name

[Show source in distribution_strategies.py:27](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L27)

Return the type of the distribution strategy.

#### Signature

```python
def get_name(self) -> str: ...
```

### DistributionStrategy().get_radius

[Show source in distribution_strategies.py:84](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L84)

Calculates the radius of the particles.

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `density` - The density of the particles.

#### Returns

- `NDArray[np.float64]` - The radius of the particles.

#### Signature

```python
@abstractmethod
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### DistributionStrategy().get_species_mass

[Show source in distribution_strategies.py:31](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L31)

The mass per species in the particles (or bin).

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `density` - The density of the particles.

#### Returns

- `NDArray[np.float64]` - The mass of the particles

#### Signature

```python
@abstractmethod
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### DistributionStrategy().get_total_mass

[Show source in distribution_strategies.py:63](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L63)

Calculates the total mass of all particles (or bin).

#### Arguments

- `distribution` - The distribution of particle sizes or masses.
- `concentration` - The concentration of each particle size or mass in
the distribution.
- `density` - The density of the particles.

#### Returns

- `np.float64` - The total mass of the particles.

#### Signature

```python
def get_total_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
) -> np.float64: ...
```



## MassBasedMovingBin

[Show source in distribution_strategies.py:165](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L165)

A strategy for particles represented by their mass distribution.

This strategy calculates particle mass, radius, and total mass based on
the particle's mass, number concentration, and density. It also moves the
bins when adding mass to the distribution.

#### Signature

```python
class MassBasedMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### MassBasedMovingBin().add_concentration

[Show source in distribution_strategies.py:197](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L197)

#### Signature

```python
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### MassBasedMovingBin().add_mass

[Show source in distribution_strategies.py:187](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L187)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### MassBasedMovingBin().collide_pairs

[Show source in distribution_strategies.py:234](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L234)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### MassBasedMovingBin().get_radius

[Show source in distribution_strategies.py:179](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L179)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### MassBasedMovingBin().get_species_mass

[Show source in distribution_strategies.py:173](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L173)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## ParticleResolvedSpeciatedMass

[Show source in distribution_strategies.py:432](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L432)

Strategy for resolved particles via speciated mass.

Strategy for resolved particles with speciated mass.
Particles may have different densities and their mass is
distributed across different species. This strategy calculates mass,
radius, and total mass based on the species at each mass, density,
the particle concentration.

#### Signature

```python
class ParticleResolvedSpeciatedMass(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### ParticleResolvedSpeciatedMass().add_concentration

[Show source in distribution_strategies.py:479](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L479)

#### Signature

```python
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### ParticleResolvedSpeciatedMass().add_mass

[Show source in distribution_strategies.py:457](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L457)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### ParticleResolvedSpeciatedMass().collide_pairs

[Show source in distribution_strategies.py:541](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L541)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### ParticleResolvedSpeciatedMass().get_radius

[Show source in distribution_strategies.py:447](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L447)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### ParticleResolvedSpeciatedMass().get_species_mass

[Show source in distribution_strategies.py:442](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L442)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## RadiiBasedMovingBin

[Show source in distribution_strategies.py:249](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L249)

A strategy for particles represented by their radius.

This strategy calculates particle mass, radius, and total mass based on
the particle's radius, number concentration, and density.

#### Signature

```python
class RadiiBasedMovingBin(DistributionStrategy): ...
```

#### See also

- [DistributionStrategy](#distributionstrategy)

### RadiiBasedMovingBin().add_concentration

[Show source in distribution_strategies.py:288](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L288)

#### Signature

```python
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### RadiiBasedMovingBin().add_mass

[Show source in distribution_strategies.py:270](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L270)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### RadiiBasedMovingBin().collide_pairs

[Show source in distribution_strategies.py:323](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L323)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### RadiiBasedMovingBin().get_radius

[Show source in distribution_strategies.py:263](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L263)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### RadiiBasedMovingBin().get_species_mass

[Show source in distribution_strategies.py:256](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L256)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## SpeciatedMassMovingBin

[Show source in distribution_strategies.py:338](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L338)

Strategy for particles with speciated mass distribution.

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

### SpeciatedMassMovingBin().add_concentration

[Show source in distribution_strategies.py:382](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L382)

#### Signature

```python
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### SpeciatedMassMovingBin().add_mass

[Show source in distribution_strategies.py:360](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L360)

#### Signature

```python
def add_mass(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    added_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### SpeciatedMassMovingBin().collide_pairs

[Show source in distribution_strategies.py:417](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L417)

#### Signature

```python
def collide_pairs(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    density: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```

### SpeciatedMassMovingBin().get_radius

[Show source in distribution_strategies.py:353](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L353)

#### Signature

```python
def get_radius(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```

### SpeciatedMassMovingBin().get_species_mass

[Show source in distribution_strategies.py:348](https://github.com/uncscode/particula/blob/main/particula/next/particles/distribution_strategies.py#L348)

#### Signature

```python
def get_species_mass(
    self, distribution: NDArray[np.float64], density: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```
