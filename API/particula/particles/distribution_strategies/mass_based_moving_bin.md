# particula.particles.distribution_strategies.mass_based_moving_bin

*module*

Mass-based moving bin distribution strategy.

This module defines :class:`MassBasedMovingBin`, a
:class:`~.base.DistributionStrategy` implementation that keeps total mass per
bin and replicates species-aware interfaces when the underlying data is not
per-species.

> **Example**
>
> >>> from particula.particles.distribution_strategies import (
> ...     MassBasedMovingBin,
> ... )
> >>> strategy = MassBasedMovingBin()
> >>> strategy.get_name()
> "MassBasedMovingBin"

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/mass_based_moving_bin.py#L1-L1)

## particula.particles.distribution_strategies.mass_based_moving_bin.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/mass_based_moving_bin.py#L25-L25)

## particula.particles.distribution_strategies.mass_based_moving_bin.MassBasedMovingBin

*class*

```python
class MassBasedMovingBin(DistributionStrategy)
```

Bases: `particula.particles.distribution_strategies.base.DistributionStrategy`

Mass-based moving bin distribution strategy.

This strategy stores the total mass per bin but exposes per-species
interfaces to downstream consumers that expect (n_particles, n_species)
inputs. Mass additions and concentration updates always operate on the
bin-level mass, while helpers replicate or average values when species are
involved.

**Attributes**

- `logger` — Module-scoped logger that records strategy warnings.

> **Example**
>
> >>> strategy = MassBasedMovingBin()
> >>> strategy.get_name()
> "MassBasedMovingBin"

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/mass_based_moving_bin.py#L28-L258)

### particula.particles.distribution_strategies.mass_based_moving_bin.MassBasedMovingBin.add_concentration

*method*

```python
def add_concentration(distribution: NDArray[np.float64], concentration: NDArray[np.float64], added_distribution: NDArray[np.float64], added_concentration: NDArray[np.float64], charge: Optional[NDArray[np.float64]] = None, added_charge: Optional[NDArray[np.float64]] = None) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Add concentration to the bins while optionally averaging charge.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — Current mass distribution per bin.
- `concentration` (`NDArray[np.float64]`) — Existing concentration per bin.
- `added_distribution` (`NDArray[np.float64]`) — Mass distribution for the added material.
- `added_concentration` (`NDArray[np.float64]`) — Concentration associated with the addition.
- `charge` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional charge to average over concentration-weighted bins.
- `added_charge` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional charge attached to the added material.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]`) — A tuple of updated distribution, concentration, and charge arrays.

**Raises**

- `ValueError` — When distribution or concentration shapes mismatch the
added arrays, or when charge shapes disagree with concentration.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/mass_based_moving_bin.py#L137-L229)

### particula.particles.distribution_strategies.mass_based_moving_bin.MassBasedMovingBin.add_mass

*method*

```python
def add_mass(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], added_mass: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Add mass to the distribution bins and keep the concentration.

This strategy increases the total mass per bin, leaving concentration
untouched because the moving bin already represents mass-based
concentrations.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — Existing mass distribution per bin.
- `concentration` (`NDArray[np.float64]`) — Particle number concentration per bin.
- `density` (`NDArray[np.float64]`) — Species density array (unused but included for interface compatibility).
- `added_mass` (`NDArray[np.float64]`) — Mass increments to add to each bin.

**Returns**

- (`NDArray[np.float64]`) — A tuple containing the updated distribution and concentration
- (`NDArray[np.float64]`) — arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/mass_based_moving_bin.py#L111-L135)

### particula.particles.distribution_strategies.mass_based_moving_bin.MassBasedMovingBin.collide_pairs

*method*

```python
def collide_pairs(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], indices: NDArray[np.int64], charge: Optional[NDArray[np.float64]] = None) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Indicate that pairwise collisions are unsupported.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — Mass distribution array.
- `concentration` (`NDArray[np.float64]`) — Concentration array for the bins.
- `density` (`NDArray[np.float64]`) — Density array used for radius/mass conversions.
- `indices` (`NDArray[np.int64]`) — Pairwise collision indices of shape (K, 2).
- `charge` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional charge array involved in the collision.

**Raises**

- `NotImplementedError` — Always raised since MassBasedMovingBin does not
represent discrete particles but aggregate mass bins.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/mass_based_moving_bin.py#L231-L258)

### particula.particles.distribution_strategies.mass_based_moving_bin.MassBasedMovingBin.get_mass

*method*

```python
def get_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Return the mass for each particle in the bin strategy.

When the distribution is one-dimensional, each entry already represents
the per-particle mass and can be returned directly. If a second axis
exists, delegate to :meth:`DistributionStrategy.get_mass` to sum across
species and preserve existing behavior.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — Mass distribution array that may include species.
- `density` (`NDArray[np.float64]`) — Density array used by the base implementation when a species axis must be collapsed.

**Returns**

- (`NDArray[np.float64]`) — A one-dimensional array of mass per particle values.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/mass_based_moving_bin.py#L72-L93)

### particula.particles.distribution_strategies.mass_based_moving_bin.MassBasedMovingBin.get_radius

*method*

```python
def get_radius(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate particle radius from individual mass values.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — Mass distribution array used to infer volume.
- `density` (`NDArray[np.float64]`) — Density array used to convert mass to volume.

**Returns**

- (`NDArray[np.float64]`) — Particle radius array in meters computed from the mass-to-volume
- (`NDArray[np.float64]`) — relationship.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/mass_based_moving_bin.py#L95-L109)

### particula.particles.distribution_strategies.mass_based_moving_bin.MassBasedMovingBin.get_species_mass

*method*

```python
def get_species_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the mass assigned to each species.

The mass-based strategy stores the total mass per bin. When the density
array includes multiple species, the total mass is duplicated across
species so hungry consumers receive (n_particles, n_species) arrays.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — Total mass per bin array with either one or two axes.
- `density` (`NDArray[np.float64]`) — Species density array determining the replicating axis when the species count exceeds one.

**Returns**

- (`NDArray[np.float64]`) — An array shaped (n_particles, n_species) representing per-species
- (`NDArray[np.float64]`) — mass values.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/mass_based_moving_bin.py#L46-L70)

### particula.particles.distribution_strategies.mass_based_moving_bin.MassBasedMovingBin.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.distribution_strategies.base.DistributionStrategy`

Return the type of the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L31-L33)

### particula.particles.distribution_strategies.mass_based_moving_bin.MassBasedMovingBin.get_total_mass

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
