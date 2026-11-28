# particula.particles.particle_data

*module*

Provide a batched particle data container for multi-box CFD simulations.

ParticleData isolates per-particle arrays from behavior while embedding the
batch dimension required for CFD experiments spanning multiple boxes.

> **Example**
>
> Single-box simulation (n_boxes=1)::
> 
>     from particula.particles.particle_data import ParticleData
>     import numpy as np
> 
>     data = ParticleData(
>         masses=np.random.rand(1, 1000, 3) * 1e-18,  # 1000 particles
>         population=np.ones((1, 1000)),
>         charge=np.zeros((1, 1000)),
>         density=np.array([1000.0, 1200.0, 800.0]),
>         volume=np.array([1e-6]),  # 1 cm^3
>         distribution_type="particle_resolved",
>     )
> 
> Multi-box CFD simulation (100 boxes)::
> 
>     cfd_data = ParticleData(
>         masses=np.zeros((100, 10000, 3)),
>         population=np.ones((100, 10000)),
>         charge=np.zeros((100, 10000)),
>         density=np.array([1000.0, 1200.0, 800.0]),
>         volume=np.ones(100) * 1e-6,
>         distribution_type="particle_resolved",
>     )

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L1-L1)

## particula.particles.particle_data.ParticleData

*class* · *dataclass*

```python
class ParticleData
```

Batched particle data container for multi-box simulations.

Simple data container with batch dimension built-in. All per-particle
arrays have shape (n_boxes, n_particles, ...) to support multi-box CFD.
Single-box simulations use n_boxes=1.

This is NOT a frozen dataclass - arrays can be updated in place for
performance in tight simulation loops. Use copy() if immutability needed.

**Attributes**

- `masses` (`NDArray[np.float64]`) — Per-species masses in kg. Shape: (n_boxes, n_particles, n_species)
- `population` (`NDArray[np.float64]`) — Extensive weights of shape (n_boxes, n_particles). Bins and resolved slots store counts. Radius PDFs store dN/dr in m^-1.
- `charge` (`NDArray[np.float64]`) — Particle charges (dimensionless integer counts). Shape: (n_boxes, n_particles)
- `density` (`NDArray[np.float64]`) — Material densities in kg/m^3. Shape: (n_species,) - shared across all boxes
- `volume` (`NDArray[np.float64]`) — Simulation volume per box in m^3, shape (n_boxes,). All tagged populations use finite positive air volumes.
- `distribution_type` (`str | None`) — Explicit storage interpretation, or ``None`` for untagged legacy carriers. New bulk helpers reject untagged data.
- `radius_grid` (`NDArray[np.float64] | None`) — Explicit PDF radius nodes in metres, shape (n_particles,); absent for other kinds.

**Raises**

- `ValueError` — If array shapes are inconsistent.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L59-L525)

### particula.particles.particle_data.ParticleData.charge

*attribute* · *instance attribute*

```python
charge: NDArray[np.float64] = charge
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L115-L115)

### particula.particles.particle_data.ParticleData.density

*attribute* · *instance attribute*

```python
density: NDArray[np.float64] = density
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L116-L116)

### particula.particles.particle_data.ParticleData.distribution_type

*attribute* · *class attribute* · *instance attribute*

```python
distribution_type: str | None = distribution_type
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L118-L118)

### particula.particles.particle_data.ParticleData.masses

*attribute* · *instance attribute*

```python
masses: NDArray[np.float64] = masses
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L114-L114)

### particula.particles.particle_data.ParticleData.population

*attribute* · *instance attribute*

```python
population: NDArray[np.float64] = population
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L127-L127)

### particula.particles.particle_data.ParticleData.radius_grid

*attribute* · *class attribute* · *instance attribute*

```python
radius_grid: NDArray[np.float64] | None = radius_grid
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L119-L119)

### particula.particles.particle_data.ParticleData.volume

*attribute* · *instance attribute*

```python
volume: NDArray[np.float64] = volume
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L117-L117)

### particula.particles.particle_data.ParticleData.concentration

*property* · *writable*

```python
concentration: NDArray[np.float64]
```

Deprecated writable 0.3 alias for legacy-compatible volumes.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L148-L151)

### particula.particles.particle_data.ParticleData.effective_density

*property*

```python
effective_density: NDArray[np.float64]
```

Mixture density from total mass divided by additive material volume.

**Returns**

- (`NDArray[np.float64]`) — Effective density in kg/m^3 with shape (n_boxes, n_particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L478-L491)

### particula.particles.particle_data.ParticleData.mass_fractions

*property*

```python
mass_fractions: NDArray[np.float64]
```

Mass fractions per species for each particle.

**Returns**

- (`NDArray[np.float64]`) — Mass fractions with shape (n_boxes, n_particles, n_species).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L494-L506)

### particula.particles.particle_data.ParticleData.n_boxes

*property*

```python
n_boxes: int
```

Number of simulation boxes.

**Returns**

- (`int`) — The size of the batch dimension (n_boxes).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L222-L228)

### particula.particles.particle_data.ParticleData.n_particles

*property*

```python
n_particles: int
```

Number of particles per box.

**Returns**

- (`int`) — The number of particles (n_particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L231-L237)

### particula.particles.particle_data.ParticleData.n_species

*property*

```python
n_species: int
```

Number of chemical species.

**Returns**

- (`int`) — The number of species (n_species).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L240-L246)

### particula.particles.particle_data.ParticleData.number_density

*property*

```python
number_density: NDArray[np.float64]
```

Return fresh per-box number density in m^-3.

Sum discrete/resolved slot densities; integrate PDF node density
over the explicit radius grid using trapezoidal quadrature.

**Returns**

- (`NDArray[np.float64]`) — Number density with shape (n_boxes,).

**Raises**

- `ValueError` — If the representation fails read-only validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L414-L432)

### particula.particles.particle_data.ParticleData.radii

*property*

```python
radii: NDArray[np.float64]
```

Particle radii derived from mass and density.

**Returns**

- (`NDArray[np.float64]`) — Radii in meters with shape (n_boxes, n_particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L249-L258)

### particula.particles.particle_data.ParticleData.slot_concentration_density

*property*

```python
slot_concentration_density: NDArray[np.float64]
```

Return fresh per-slot number density or radius-PDF density.

Discrete and resolved slots have units m^-3; PDF nodes have units
m^-4. The result has shape (n_boxes, n_particles) and is detached
from the raw population array.

**Returns**

- (`NDArray[np.float64]`) — Per-slot density on the declared representation's grid.

**Raises**

- `ValueError` — If the kind, grid, volume, or raw data is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L337-L351)

### particula.particles.particle_data.ParticleData.species_mass_density

*property*

```python
species_mass_density: NDArray[np.float64]
```

Return fresh per-species particle mass density in kg/m^3.

Weight species masses by raw slot density. For PDFs, integrate
the product of mass and dN/dr on the explicit radius grid using
trapezoidal node weights, rather than summing PDF nodes.

**Returns**

- (`NDArray[np.float64]`) — Particle mass density with shape (n_boxes, n_species).

**Raises**

- `ValueError` — If the representation fails read-only validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L435-L457)

### particula.particles.particle_data.ParticleData.species_mass_inventory

*property*

```python
species_mass_inventory: NDArray[np.float64]
```

Return fresh extensive particle mass per species in kg.

**Returns**

- (`NDArray[np.float64]`) — Mass inventory with shape (n_boxes, n_species).

**Raises**

- `ValueError` — If the representation fails read-only validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L460-L475)

### particula.particles.particle_data.ParticleData.total_mass

*property*

```python
total_mass: NDArray[np.float64]
```

Total mass per particle.

**Returns**

- (`NDArray[np.float64]`) — Total mass in kilograms with shape (n_boxes, n_particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L261-L267)

### particula.particles.particle_data.ParticleData.__init__

*method*

```python
def __init__(masses: NDArray[np.float64], concentration: NDArray[np.float64] | None = None, charge: NDArray[np.float64] | None = None, density: NDArray[np.float64] | None = None, volume: NDArray[np.float64] | None = None, distribution_type: str | None = None, radius_grid: NDArray[np.float64] | None = None, *, population: NDArray[np.float64] | None = None) -> None
```

Bind extensive storage or the compatible 0.3 concentration array.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L99-L134)

### particula.particles.particle_data.ParticleData.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate raw array shapes without interpreting distribution kind.

**Raises**

- `ValueError` — If raw field shapes or species dimensions disagree.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L160-L219)

### particula.particles.particle_data.ParticleData._check_legacy_population

*method*

```python
def _check_legacy_population() -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L136-L144)

### particula.particles.particle_data.ParticleData.copy

*method*

```python
def copy() -> ParticleData
```

Create a deep copy of this ParticleData.

**Returns**

- (`ParticleData`) — A new ParticleData with detached raw arrays and radius grid,
- (`ParticleData`) — preserving the explicit distribution type when present.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L508-L525)

### particula.particles.particle_data.ParticleData.set_slot_concentration_density

*method*

```python
def set_slot_concentration_density(density_by_slot: ArrayLike, *, volume: ArrayLike | None = None) -> None
```

Replace particle population from per-slot density and air volume.

**Parameters**

- `density_by_slot` (`ArrayLike`) — Nonnegative number densities (m^-3) for bins or resolved slots, or radius-PDF ordinates (m^-4). Supply shape (n_boxes, n_particles), or (n_particles,) for one box.
- `volume` (`ArrayLike | None`) (default: `None`) — Represented air volume (m^3), as one positive scalar for all boxes or an array of shape (n_boxes,). Omit to use each box's current volume.

Writes density times the selected volume into the existing population
array. A supplied volume is committed in the same update.

**Raises**

- `TypeError` — If an input does not contain real numbers.
- `ValueError` — If the representation, shape, values or storage
prevent an in-place update. Both arrays retain their values.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L353-L411)

### particula.particles.particle_data.ParticleData.validate_representation

*method*

```python
def validate_representation() -> None
```

Validate storage for its declared distribution interpretation.

Call after directly changing writable arrays. This read-only check
never repairs state; derived population helpers call it on every read.
Each box requires positive represented volume. PDF nodes must form an
increasing, positive radius grid with at least two nodes.

**Raises**

- `ValueError` — If kind, grid, volume, shape, or values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L269-L334)

## particula.particles.particle_data._replacement_volumes

*function*

```python
def _replacement_volumes(volume: ArrayLike, shape: tuple[int, ...]) -> NDArray[np.float64]
```

Stage an independent volume value for each box before updating state.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L43-L56)
