# particula.particles.distribution_strategies.speciated_mass_moving_bin

*module*

Speciated mass moving bin strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L1-L1)

## particula.particles.distribution_strategies.speciated_mass_moving_bin.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L11-L11)

## particula.particles.distribution_strategies.speciated_mass_moving_bin.SpeciatedMassMovingBin

*class*

```python
class SpeciatedMassMovingBin(DistributionStrategy)
```

Bases: `particula.particles.distribution_strategies.base.DistributionStrategy`

Strategy for particles with speciated mass distribution.

Each particle may contain multiple species, each with a unique
density. This strategy calculates mass, radius, and total mass from
the species-level masses and overall particle concentrations.

Methods:
- get_name : Return the type of the distribution strategy.
- get_species_mass : Calculate the mass per species.
- get_mass : Calculate the mass of the particles or bin.
- get_total_mass : Calculate the total mass of particles.
- get_radius : Calculate the radius of particles.
- add_mass : Add mass to the particle distribution.
- add_concentration : Add concentration to the distribution.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L14-L201)

### particula.particles.distribution_strategies.speciated_mass_moving_bin.SpeciatedMassMovingBin.add_concentration

*method*

```python
def add_concentration(distribution: NDArray[np.float64], concentration: NDArray[np.float64], added_distribution: NDArray[np.float64], added_concentration: NDArray[np.float64], charge: Optional[NDArray[np.float64]] = None, added_charge: Optional[NDArray[np.float64]] = None) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Add concentration to the distribution with optional charge.

Charge is updated using concentration-weighted averaging when both
``charge`` and ``added_charge`` are supplied. If ``charge`` is ``None``
it is returned as ``None``. When ``added_charge`` is ``None`` the
existing charge is preserved. Empty bins fall back to ``added_charge``
to avoid divide-by-zero. Distribution mismatches on shape or value
raise ``ValueError`` to mirror sibling strategies.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]`) — Updated distribution, concentration, and charge arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L81-L168)

### particula.particles.distribution_strategies.speciated_mass_moving_bin.SpeciatedMassMovingBin.add_mass

*method*

```python
def add_mass(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], added_mass: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Add mass to the particle distribution per species.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64]]`) — Updated distribution and concentration arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L59-L79)

### particula.particles.distribution_strategies.speciated_mass_moving_bin.SpeciatedMassMovingBin.collide_pairs

*method*

```python
def collide_pairs(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], indices: NDArray[np.int64], charge: Optional[NDArray[np.float64]] = None) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Collide particle pairs (not implemented for this strategy).

This method is not implemented for SpeciatedMassMovingBin because
particle pair collisions are not physically valid for bin-based
strategies where particles are represented by fixed mass bins with
concentrations.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The speciated mass distribution array.
- `concentration` (`NDArray[np.float64]`) — The concentration array.
- `density` (`NDArray[np.float64]`) — The density array.
- `indices` (`NDArray[np.int64]`) — Collision pair indices array of shape (K, 2).
- `charge` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional charge array (unused in this strategy).

**Raises**

- `NotImplementedError` — Always raised as method is not applicable.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L170-L201)

### particula.particles.distribution_strategies.speciated_mass_moving_bin.SpeciatedMassMovingBin.get_radius

*method*

```python
def get_radius(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate particle radius from multi-species mass and density.

**Returns**

- (`NDArray[np.float64]`) — Particle radius in meters.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L41-L57)

### particula.particles.distribution_strategies.speciated_mass_moving_bin.SpeciatedMassMovingBin.get_species_mass

*method*

```python
def get_species_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the mass per species for each bin.

**Returns**

- (`NDArray[np.float64]`) — Mass per species array.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L31-L39)

### particula.particles.distribution_strategies.speciated_mass_moving_bin.SpeciatedMassMovingBin.get_mass

*method*

```python
def get_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Inherited from: `particula.particles.distribution_strategies.base.DistributionStrategy`

Calculate the mass of the particles or bin.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The distribution of particle sizes or masses.
- `density` (`NDArray[np.float64]`) — The density of the particles.

**Returns**

- (`NDArray[np.float64]`) — The mass of the particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L49-L63)

### particula.particles.distribution_strategies.speciated_mass_moving_bin.SpeciatedMassMovingBin.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.distribution_strategies.base.DistributionStrategy`

Return the type of the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L31-L33)

### particula.particles.distribution_strategies.speciated_mass_moving_bin.SpeciatedMassMovingBin.get_total_mass

*method*

```python
def get_total_mass(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64]) -> np.float64
```

Inherited from: `particula.particles.distribution_strategies.base.DistributionStrategy`

Calculate the total mass of all particles (or bin).

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The distribution of particle sizes or masses.
- `concentration` (`NDArray[np.float64]`) — The concentration of each particle size or mass in the distribution.
- `density` (`NDArray[np.float64]`) — The density of the particles.

**Returns**

- (`np.float64`) — The total mass of the particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L65-L83)
