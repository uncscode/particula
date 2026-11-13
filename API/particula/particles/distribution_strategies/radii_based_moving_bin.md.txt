# particula.particles.distribution_strategies.radii_based_moving_bin

*module*

Radii-based moving bin strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L1-L1)

## particula.particles.distribution_strategies.radii_based_moving_bin.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L11-L11)

## particula.particles.distribution_strategies.radii_based_moving_bin.RadiiBasedMovingBin

*class*

```python
class RadiiBasedMovingBin(DistributionStrategy)
```

Bases: `particula.particles.distribution_strategies.base.DistributionStrategy`

Strategy for particles represented by their radius distribution.

Calculates particle mass, radius, and total mass based on particle
radius, number concentration, and density. This moving-bin approach
recalculates radii when mass is added.

Methods:
- get_name : Return the type of the distribution strategy.
- get_species_mass : Calculate the mass per species.
- get_mass : Calculate the mass of the particles or bin.
- get_total_mass : Calculate the total mass of particles.
- get_radius : Calculate the radius of particles.
- add_mass : Add mass to the particle distribution.
- add_concentration : Add concentration to the distribution.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L14-L195)

### particula.particles.distribution_strategies.radii_based_moving_bin.RadiiBasedMovingBin.add_concentration

*method*

```python
def add_concentration(distribution: NDArray[np.float64], concentration: NDArray[np.float64], added_distribution: NDArray[np.float64], added_concentration: NDArray[np.float64], charge: Optional[NDArray[np.float64]] = None, added_charge: Optional[NDArray[np.float64]] = None) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Add concentration to the distribution with optional charge.

Charge is updated using concentration-weighted averaging when both
``charge`` and ``added_charge`` are supplied. If ``charge`` is ``None``
it is returned as ``None``. When ``added_charge`` is ``None`` the
existing charge is preserved. Empty bins fall back to ``added_charge``
to avoid divide-by-zero.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]`) — Updated distribution, concentration, and charge arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L78-L164)

### particula.particles.distribution_strategies.radii_based_moving_bin.RadiiBasedMovingBin.add_mass

*method*

```python
def add_mass(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], added_mass: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Add mass to the particle distribution and update radii.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64]]`) — Updated distribution (radii) and concentration arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L52-L76)

### particula.particles.distribution_strategies.radii_based_moving_bin.RadiiBasedMovingBin.collide_pairs

*method*

```python
def collide_pairs(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], indices: NDArray[np.int64], charge: Optional[NDArray[np.float64]] = None) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Collide particle pairs (not implemented for this strategy).

This method is not implemented for RadiiBasedMovingBin because particle
pair collisions are not physically valid for bin-based strategies
where particles are represented by fixed radius bins with
concentrations.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The radius distribution array.
- `concentration` (`NDArray[np.float64]`) — The concentration array.
- `density` (`NDArray[np.float64]`) — The density array.
- `indices` (`NDArray[np.int64]`) — Collision pair indices array of shape (K, 2).
- `charge` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional charge array (unused in this strategy).

**Raises**

- `NotImplementedError` — Always raised as method is not applicable.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L166-L195)

### particula.particles.distribution_strategies.radii_based_moving_bin.RadiiBasedMovingBin.get_radius

*method*

```python
def get_radius(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Return particle radius from the distribution.

**Returns**

- (`NDArray[np.float64]`) — Particle radius in meters.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L42-L50)

### particula.particles.distribution_strategies.radii_based_moving_bin.RadiiBasedMovingBin.get_species_mass

*method*

```python
def get_species_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the mass per species from radius and density.

**Returns**

- (`NDArray[np.float64]`) — Mass per species array.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L31-L40)

### particula.particles.distribution_strategies.radii_based_moving_bin.RadiiBasedMovingBin.get_mass

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

### particula.particles.distribution_strategies.radii_based_moving_bin.RadiiBasedMovingBin.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.distribution_strategies.base.DistributionStrategy`

Return the type of the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L31-L33)

### particula.particles.distribution_strategies.radii_based_moving_bin.RadiiBasedMovingBin.get_total_mass

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
