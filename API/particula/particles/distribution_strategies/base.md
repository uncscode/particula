# particula.particles.distribution_strategies.base

*module*

Particle Distribution Strategies.

Defines various distribution strategies (mass-based, radii-based, etc.)
for representing particles in a distribution. Each strategy handles
mass, radius, total mass, and concentration updates differently.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L1-L1)

## particula.particles.distribution_strategies.base.DistributionStrategy

*class*

```python
class DistributionStrategy(ABC)
```

Bases: `ABC`

Abstract base class defining common interfaces for
mass, radius, and total mass calculations across
different particle distribution representations.

Methods:
- get_name : Return the type of the distribution strategy.
- get_species_mass : Calculate the mass per species.
- get_mass : Calculate the mass of the particles or bin.
- get_total_mass : Calculate the total mass of particles.
- get_radius : Calculate the radius of particles.
- add_mass : Add mass to the particle distribution.
- add_concentration : Add concentration to the distribution.
- collide_pairs : Perform collision logic on specified particle pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L15-L186)

### particula.particles.distribution_strategies.base.DistributionStrategy.add_concentration

*method* · *abstract*

```python
def add_concentration(distribution: NDArray[np.float64], concentration: NDArray[np.float64], added_distribution: NDArray[np.float64], added_concentration: NDArray[np.float64], charge: Optional[NDArray[np.float64]] = None, added_charge: Optional[NDArray[np.float64]] = None) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Add concentration to the distribution of particles.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The distribution of particle sizes or masses.
- `concentration` (`NDArray[np.float64]`) — The concentration of each particle size or mass.
- `added_distribution` (`NDArray[np.float64]`) — The distribution to be added.
- `added_concentration` (`NDArray[np.float64]`) — The concentration to be added.
- `charge` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional charge array to update alongside concentration. If None, charge is ignored and returned None.
- `added_charge` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional charge array for added particles.

**Returns**

- (`NDArray[np.float64]`) — The updated distribution array.
- (`NDArray[np.float64]`) — The updated concentration array.
- (`Optional[NDArray[np.float64]]`) — The updated charge array, or None if charge input was None.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L121-L151)

### particula.particles.distribution_strategies.base.DistributionStrategy.add_mass

*method* · *abstract*

```python
def add_mass(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], added_mass: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Add mass to the distribution of particles.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The distribution of particle sizes or masses.
- `concentration` (`NDArray[np.float64]`) — The concentration of each particle size or mass.
- `density` (`NDArray[np.float64]`) — The density of the particles.
- `added_mass` (`NDArray[np.float64]`) — The mass to be added per distribution bin.

**Returns**

- (`NDArray[np.float64]`) — The updated distribution array.
- (`NDArray[np.float64]`) — The updated concentration array.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L99-L119)

### particula.particles.distribution_strategies.base.DistributionStrategy.collide_pairs

*method* · *abstract*

```python
def collide_pairs(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], indices: NDArray[np.int64], charge: Optional[NDArray[np.float64]] = None) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Collide index pairs in the distribution.

Performs collision between particle pairs specified by indices. The
smaller particle (first index in each pair) is merged into the larger
particle (second index), with mass being summed and concentration of
the smaller particle set to zero.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The distribution of particle sizes or masses.
- `concentration` (`NDArray[np.float64]`) — The concentration of each particle size or mass.
- `density` (`NDArray[np.float64]`) — The density of the particles.
- `indices` (`NDArray[np.int64]`) — The indices of the particle pairs to collide, shape (N, 2) where each row is [small_index, large_index].
- `charge` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional charge array. If provided and contains non-zero values in colliding pairs, charges will be summed during collisions. If None, charge handling is skipped.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]`) — A tuple containing: - Updated distribution array with merged masses. - Updated concentration array with zeroed small particles. - Updated charge array (None if input was None).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L153-L186)

### particula.particles.distribution_strategies.base.DistributionStrategy.get_mass

*method*

```python
def get_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the mass of the particles or bin.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The distribution of particle sizes or masses.
- `density` (`NDArray[np.float64]`) — The density of the particles.

**Returns**

- (`NDArray[np.float64]`) — The mass of the particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L49-L63)

### particula.particles.distribution_strategies.base.DistributionStrategy.get_name

*method*

```python
def get_name() -> str
```

Return the type of the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L31-L33)

### particula.particles.distribution_strategies.base.DistributionStrategy.get_radius

*method* · *abstract*

```python
def get_radius(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the radius of the particles.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The distribution of particle sizes or masses.
- `density` (`NDArray[np.float64]`) — The density of the particles.

**Returns**

- (`NDArray[np.float64]`) — The radius of the particles in meters.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L85-L97)

### particula.particles.distribution_strategies.base.DistributionStrategy.get_species_mass

*method* · *abstract*

```python
def get_species_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Return the mass per species in the distribution.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The distribution of particle sizes or masses.
- `density` (`NDArray[np.float64]`) — The density of the particles.

**Returns**

- (`NDArray[np.float64]`) — The mass of the particles (per species).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L35-L47)

### particula.particles.distribution_strategies.base.DistributionStrategy.get_total_mass

*method*

```python
def get_total_mass(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64]) -> np.float64
```

Calculate the total mass of all particles (or bin).

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The distribution of particle sizes or masses.
- `concentration` (`NDArray[np.float64]`) — The concentration of each particle size or mass in the distribution.
- `density` (`NDArray[np.float64]`) — The density of the particles.

**Returns**

- (`np.float64`) — The total mass of the particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L65-L83)
