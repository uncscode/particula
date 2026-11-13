# particula.particles.distribution_strategies.particle_resolved_speciated_mass

*module*

Particle resolved speciated mass strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L1-L1)

## particula.particles.distribution_strategies.particle_resolved_speciated_mass.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L11-L11)

## particula.particles.distribution_strategies.particle_resolved_speciated_mass.ParticleResolvedSpeciatedMass

*class*

```python
class ParticleResolvedSpeciatedMass(DistributionStrategy)
```

Bases: `particula.particles.distribution_strategies.base.DistributionStrategy`

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

### particula.particles.distribution_strategies.particle_resolved_speciated_mass.ParticleResolvedSpeciatedMass.add_concentration

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

### particula.particles.distribution_strategies.particle_resolved_speciated_mass.ParticleResolvedSpeciatedMass.add_mass

*method*

```python
def add_mass(distribution: NDArray[np.float64], concentration: NDArray[np.float64], density: NDArray[np.float64], added_mass: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Add mass to individual particles in the distribution.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64]]`) — Updated distribution and concentration arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L56-L83)

### particula.particles.distribution_strategies.particle_resolved_speciated_mass.ParticleResolvedSpeciatedMass.collide_pairs

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

### particula.particles.distribution_strategies.particle_resolved_speciated_mass.ParticleResolvedSpeciatedMass.get_radius

*method*

```python
def get_radius(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate particle radius from multi-species mass and density.

**Returns**

- (`NDArray[np.float64]`) — Particle radius in meters for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L42-L54)

### particula.particles.distribution_strategies.particle_resolved_speciated_mass.ParticleResolvedSpeciatedMass.get_species_mass

*method*

```python
def get_species_mass(distribution: NDArray[np.float64], density: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the mass per species for each particle.

**Returns**

- (`NDArray[np.float64]`) — Mass per species array for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/particle_resolved_speciated_mass.py#L32-L40)

### particula.particles.distribution_strategies.particle_resolved_speciated_mass.ParticleResolvedSpeciatedMass.get_mass

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

### particula.particles.distribution_strategies.particle_resolved_speciated_mass.ParticleResolvedSpeciatedMass.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.distribution_strategies.base.DistributionStrategy`

Return the type of the distribution strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/distribution_strategies/base.py#L31-L33)

### particula.particles.distribution_strategies.particle_resolved_speciated_mass.ParticleResolvedSpeciatedMass.get_total_mass

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
