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
>         concentration=np.ones((1, 1000)),
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
>         concentration=np.ones((100, 10000)),
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
- `concentration` (`NDArray[np.float64]`) — Raw slot weights of shape (n_boxes, n_particles). Discrete bins store number per m^3, PDF nodes store dN/dr in m^-4, and resolved slots store counts.
- `charge` (`NDArray[np.float64]`) — Particle charges (dimensionless integer counts). Shape: (n_boxes, n_particles)
- `density` (`NDArray[np.float64]`) — Material densities in kg/m^3. Shape: (n_species,) - shared across all boxes
- `volume` (`NDArray[np.float64]`) — Simulation volume per box in m^3, shape (n_boxes,). Tagged discrete/PDF storage uses 1 m^3; tagged resolved storage uses positive physical volume.
- `distribution_type` (`str | None`) — Explicit storage interpretation, or ``None`` for untagged legacy carriers. New bulk helpers reject untagged data.
- `radius_grid` (`NDArray[np.float64] | None`) — Explicit PDF radius nodes in metres, shape (n_particles,); absent for other kinds.

**Raises**

- `ValueError` — If array shapes are inconsistent.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L39-L391)

### particula.particles.particle_data.ParticleData.charge

*attribute* · *instance attribute*

```python
charge: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L74-L74)

### particula.particles.particle_data.ParticleData.concentration

*attribute* · *instance attribute*

```python
concentration: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L73-L73)

### particula.particles.particle_data.ParticleData.density

*attribute* · *instance attribute*

```python
density: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L75-L75)

### particula.particles.particle_data.ParticleData.distribution_type

*attribute* · *class attribute* · *instance attribute*

```python
distribution_type: str | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L77-L77)

### particula.particles.particle_data.ParticleData.masses

*attribute* · *instance attribute*

```python
masses: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L72-L72)

### particula.particles.particle_data.ParticleData.radius_grid

*attribute* · *class attribute* · *instance attribute*

```python
radius_grid: NDArray[np.float64] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L78-L78)

### particula.particles.particle_data.ParticleData.volume

*attribute* · *instance attribute*

```python
volume: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L76-L76)

### particula.particles.particle_data.ParticleData.effective_density

*property*

```python
effective_density: NDArray[np.float64]
```

Mass-weighted effective density per particle.

**Returns**

- (`NDArray[np.float64]`) — Effective density in kg/m^3 with shape (n_boxes, n_particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L342-L357)

### particula.particles.particle_data.ParticleData.mass_fractions

*property*

```python
mass_fractions: NDArray[np.float64]
```

Mass fractions per species for each particle.

**Returns**

- (`NDArray[np.float64]`) — Mass fractions with shape (n_boxes, n_particles, n_species).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L360-L372)

### particula.particles.particle_data.ParticleData.n_boxes

*property*

```python
n_boxes: int
```

Number of simulation boxes.

**Returns**

- (`int`) — The size of the batch dimension (n_boxes).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L142-L148)

### particula.particles.particle_data.ParticleData.n_particles

*property*

```python
n_particles: int
```

Number of particles per box.

**Returns**

- (`int`) — The number of particles (n_particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L151-L157)

### particula.particles.particle_data.ParticleData.n_species

*property*

```python
n_species: int
```

Number of chemical species.

**Returns**

- (`int`) — The number of species (n_species).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L160-L166)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L280-L298)

### particula.particles.particle_data.ParticleData.radii

*property*

```python
radii: NDArray[np.float64]
```

Particle radii derived from mass and density.

**Returns**

- (`NDArray[np.float64]`) — Radii in meters with shape (n_boxes, n_particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L169-L178)

### particula.particles.particle_data.ParticleData.slot_concentration_density

*property*

```python
slot_concentration_density: NDArray[np.float64]
```

Return fresh per-slot number density or radius-PDF density.

Discrete and resolved slots have units m^-3; PDF nodes have units
m^-4. The result has shape (n_boxes, n_particles) and is detached
from the raw concentration array.

**Returns**

- (`NDArray[np.float64]`) — Per-slot density on the declared representation's grid.

**Raises**

- `ValueError` — If the kind, grid, volume, or raw data is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L261-L277)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L301-L327)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L330-L339)

### particula.particles.particle_data.ParticleData.total_mass

*property*

```python
total_mass: NDArray[np.float64]
```

Total mass per particle.

**Returns**

- (`NDArray[np.float64]`) — Total mass in kilograms with shape (n_boxes, n_particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L181-L187)

### particula.particles.particle_data.ParticleData.__init__

*method*

```python
def __init__(masses: NDArray[np.float64], concentration: NDArray[np.float64], charge: NDArray[np.float64], density: NDArray[np.float64], volume: NDArray[np.float64], distribution_type: str | None = None, radius_grid: NDArray[np.float64] | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L0-L0)

### particula.particles.particle_data.ParticleData.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate raw array shapes without interpreting distribution kind.

**Raises**

- `ValueError` — If raw field shapes or species dimensions disagree.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L80-L139)

### particula.particles.particle_data.ParticleData.copy

*method*

```python
def copy() -> ParticleData
```

Create a deep copy of this ParticleData.

**Returns**

- (`ParticleData`) — A new ParticleData with detached raw arrays and radius grid,
- (`ParticleData`) — preserving the explicit distribution type when present.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L374-L391)

### particula.particles.particle_data.ParticleData.validate_representation

*method*

```python
def validate_representation() -> None
```

Validate storage for its declared distribution interpretation.

Call after directly changing writable arrays. This read-only check
never repairs state; derived population helpers call it on every read.
Discrete and PDF storage requires exactly 1 m^3 per box; resolved
storage requires positive volume. PDF nodes must form an increasing,
positive radius grid with at least two nodes.

**Raises**

- `ValueError` — If kind, grid, volume, shape, or values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data.py#L189-L258)
