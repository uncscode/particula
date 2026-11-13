# particula.particles.distribution_strategies

*module*

Distribution strategy package.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/__init__.py#L1-L1)

## particula.particles.distribution_strategies.DistributionStrategy

*class*

```python
class DistributionStrategy(ABC)
```

Bases: `ABC`

Re-exported from: `particula.particles.distribution_strategies.base`

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

### particula.particles.distribution_strategies.DistributionStrategy.add_concentration

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

### particula.particles.distribution_strategies.DistributionStrategy.add_mass

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

### particula.particles.distribution_strategies.DistributionStrategy.collide_pairs

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

### particula.particles.distribution_strategies.DistributionStrategy.get_mass

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

### particula.particles.distribution_strategies.DistributionStrategy.get_name

*method*

```python
def get_name() -> str
```

Return the type of the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L31-L33)

### particula.particles.distribution_strategies.DistributionStrategy.get_radius

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

### particula.particles.distribution_strategies.DistributionStrategy.get_species_mass

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

### particula.particles.distribution_strategies.DistributionStrategy.get_total_mass

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

## particula.particles.distribution_strategies.MassBasedMovingBin

*class*

```python
class MassBasedMovingBin(DistributionStrategy)
```

Bases: `particula.particles.distribution_strategies.base.DistributionStrategy`

Re-exported from: `particula.particles.distribution_strategies.mass_based_moving_bin`

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

### particula.particles.distribution_strategies.MassBasedMovingBin.add_concentration

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

### particula.particles.distribution_strategies.MassBasedMovingBin.add_mass

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

### particula.particles.distribution_strategies.MassBasedMovingBin.collide_pairs

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

### particula.particles.distribution_strategies.MassBasedMovingBin.get_mass

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

### particula.particles.distribution_strategies.MassBasedMovingBin.get_radius

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

### particula.particles.distribution_strategies.MassBasedMovingBin.get_species_mass

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

### particula.particles.distribution_strategies.MassBasedMovingBin.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.distribution_strategies.base.DistributionStrategy`

Return the type of the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L31-L33)

### particula.particles.distribution_strategies.MassBasedMovingBin.get_total_mass

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

## particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass

*class*

```python
class ParticleResolvedSpeciatedMass(DistributionStrategy)
```

Bases: `particula.particles.distribution_strategies.base.DistributionStrategy`

Re-exported from: `particula.particles.distribution_strategies.particle_resolved_speciated_mass`

Strategy for particle-resolved masses with multiple species.

Allows each particle to have separate masses for each species, with
individualized densities. This strategy provides a more detailed
approach when each particle's composition must be modeled explicitly.

Methods:
- get_name : Return the type of the distribution strategy.
- get_species_mass : Calculate the mass per species.
- get_mass : Calculate the mass of the particles or bin.
- get_total_mass : Calculate the total mass of particles.
- get_radius : Calculate the radius of particles.
- add_mass : Add mass to the particle distribution.
- add_concentration : Add concentration to the distribution.
- collide_pairs : Perform collision logic on specified particle pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L14-L243)

### particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass.add_concentration

*method*

```python
def add_concentration(distribution: NDArray[np.float64], concentration: NDArray[np.float64], added_distribution: NDArray[np.float64], added_concentration: NDArray[np.float64], charge: Optional[NDArray[np.float64]] = None, added_charge: Optional[NDArray[np.float64]] = None) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Add new particles to the distribution with optional charge.

Charge handling mirrors the fill-then-append logic used for
concentration: empty bins are filled first, then remaining particles
are appended. Charge is only processed when a charge array is provided;
otherwise charge is passed through as None to preserve compatibility.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]`) — Updated distribution, concentration, and charge arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L85-L177)

### particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass.add_mass

*method*

```python
def add_mass(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], added_mass: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Add mass to individual particles in the distribution.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64]]`) — Updated distribution and concentration arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L56-L83)

### particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass.collide_pairs

*method*

```python
def collide_pairs(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], indices: NDArray[np.int64], charge: Optional[NDArray[np.float64]] = None) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Collide specified particle pairs by merging mass and charge.

Performs coagulation between particle pairs for particle-resolved
simulations. The smaller particle's mass is added to the larger
particle, and the smaller particle's concentration is set to zero.
If a charge array is provided, charges are conserved by summing the
charges of the colliding pair.

The charge handling is optimized: charges are only processed when the
charge array is provided as a numpy array AND at least one of the
colliding particles has a non-zero charge.

**Parameters**

- `distribution` (`NDArray[np.float64]`) — The mass distribution array. Shape is (N,) for single species or (N, M) for M species per particle.
- `concentration` (`NDArray[np.float64]`) — The concentration array of shape (N,).
- `density` (`NDArray[np.float64]`) — The density array of shape (M,) for species densities.
- `indices` (`NDArray[np.int64]`) — Collision pair indices array of shape (K, 2) where each row is [small_index, large_index].
- `charge` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional charge array of shape (N,). If provided and contains non-zero values in colliding pairs, charges will be summed during collisions. If None, charge handling is skipped.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]`) — A tuple containing: - Updated distribution array with merged masses. - Updated concentration array with zeroed small particles. - Updated charge array (None if input was None).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L179-L243)

### particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass.get_radius

*method*

```python
def get_radius(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate particle radius from multi-species mass and density.

**Returns**

- (`NDArray[np.float64]`) — Particle radius in meters for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L42-L54)

### particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass.get_species_mass

*method*

```python
def get_species_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the mass per species for each particle.

**Returns**

- (`NDArray[np.float64]`) — Mass per species array for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L32-L40)

### particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass.get_mass

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

### particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.distribution_strategies.base.DistributionStrategy`

Return the type of the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L31-L33)

### particula.particles.distribution_strategies.ParticleResolvedSpeciatedMass.get_total_mass

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

## particula.particles.distribution_strategies.RadiiBasedMovingBin

*class*

```python
class RadiiBasedMovingBin(DistributionStrategy)
```

Bases: `particula.particles.distribution_strategies.base.DistributionStrategy`

Re-exported from: `particula.particles.distribution_strategies.radii_based_moving_bin`

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

### particula.particles.distribution_strategies.RadiiBasedMovingBin.add_concentration

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

### particula.particles.distribution_strategies.RadiiBasedMovingBin.add_mass

*method*

```python
def add_mass(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], added_mass: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Add mass to the particle distribution and update radii.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64]]`) — Updated distribution (radii) and concentration arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L52-L76)

### particula.particles.distribution_strategies.RadiiBasedMovingBin.collide_pairs

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

### particula.particles.distribution_strategies.RadiiBasedMovingBin.get_radius

*method*

```python
def get_radius(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Return particle radius from the distribution.

**Returns**

- (`NDArray[np.float64]`) — Particle radius in meters.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L42-L50)

### particula.particles.distribution_strategies.RadiiBasedMovingBin.get_species_mass

*method*

```python
def get_species_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the mass per species from radius and density.

**Returns**

- (`NDArray[np.float64]`) — Mass per species array.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/radii_based_moving_bin.py#L31-L40)

### particula.particles.distribution_strategies.RadiiBasedMovingBin.get_mass

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

### particula.particles.distribution_strategies.RadiiBasedMovingBin.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.distribution_strategies.base.DistributionStrategy`

Return the type of the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L31-L33)

### particula.particles.distribution_strategies.RadiiBasedMovingBin.get_total_mass

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

## particula.particles.distribution_strategies.SpeciatedMassMovingBin

*class*

```python
class SpeciatedMassMovingBin(DistributionStrategy)
```

Bases: `particula.particles.distribution_strategies.base.DistributionStrategy`

Re-exported from: `particula.particles.distribution_strategies.speciated_mass_moving_bin`

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

### particula.particles.distribution_strategies.SpeciatedMassMovingBin.add_concentration

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

### particula.particles.distribution_strategies.SpeciatedMassMovingBin.add_mass

*method*

```python
def add_mass(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], added_mass: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Add mass to the particle distribution per species.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64]]`) — Updated distribution and concentration arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L59-L79)

### particula.particles.distribution_strategies.SpeciatedMassMovingBin.collide_pairs

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

### particula.particles.distribution_strategies.SpeciatedMassMovingBin.get_radius

*method*

```python
def get_radius(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate particle radius from multi-species mass and density.

**Returns**

- (`NDArray[np.float64]`) — Particle radius in meters.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L41-L57)

### particula.particles.distribution_strategies.SpeciatedMassMovingBin.get_species_mass

*method*

```python
def get_species_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the mass per species for each bin.

**Returns**

- (`NDArray[np.float64]`) — Mass per species array.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/speciated_mass_moving_bin.py#L31-L39)

### particula.particles.distribution_strategies.SpeciatedMassMovingBin.get_mass

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

### particula.particles.distribution_strategies.SpeciatedMassMovingBin.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.distribution_strategies.base.DistributionStrategy`

Return the type of the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L31-L33)

### particula.particles.distribution_strategies.SpeciatedMassMovingBin.get_total_mass

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

**Modules**

- `particula.particles.distribution_strategies.base`
- `particula.particles.distribution_strategies.mass_based_moving_bin`
- `particula.particles.distribution_strategies.particle_resolved_speciated_mass`
- `particula.particles.distribution_strategies.radii_based_moving_bin`
- `particula.particles.distribution_strategies.speciated_mass_moving_bin`
