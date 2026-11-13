# particula.particles.particle_data_builder

*module*

Builder for creating validated :class:`ParticleData` instances.

Provides a fluent interface to set particle fields with optional unit
conversion, automatic batch dimension handling, and a zero-initialization
path for cases where masses are not precomputed.

**Examples**

Single-box with explicit masses::

    import numpy as np
    from particula.particles import ParticleDataBuilder

    data = (
        ParticleDataBuilder()
        .set_distribution_type("discrete")
        .set_masses(np.array([[1e-18, 2e-18]]), units="kg")
        .set_density(np.array([1000.0, 1200.0]), units="kg/m^3")
        .set_concentration(np.array([1.0]), units="1/m^3")
        .set_charge(np.array([0.0]))
        .build()
    )

Zero-init with counts only::

    data = (
        ParticleDataBuilder()
        .set_distribution_type("discrete")
        .set_n_boxes(2)
        .set_n_particles(3)
        .set_n_species(1)
        .set_density(np.array([1500.0]), units="kg/m^3")
        .build()
    )

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L1-L1)

## particula.particles.particle_data_builder.ParticleDataBuilder

*class*

```python
class ParticleDataBuilder
```

Fluent builder that prepares arrays for ``ParticleData``.

Each setter performs a single unit conversion and ensures ``float64``
dtype. Batch dimensions are inserted automatically when 1D (for
concentration/charge) or 2D (for masses). A zero-initialization path is
available when dimensions are supplied instead of masses. Building
requires an explicit storage kind; PDF storage also requires a radius
grid and matching PDF concentration units.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L47-L407)

### particula.particles.particle_data_builder.ParticleDataBuilder._charge

*attribute* · *instance attribute*

```python
_charge: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L62-L62)

### particula.particles.particle_data_builder.ParticleDataBuilder._concentration

*attribute* · *instance attribute*

```python
_concentration: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L61-L61)

### particula.particles.particle_data_builder.ParticleDataBuilder._concentration_units

*attribute* · *instance attribute*

```python
_concentration_units: str | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L70-L70)

### particula.particles.particle_data_builder.ParticleDataBuilder._density

*attribute* · *instance attribute*

```python
_density: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L63-L63)

### particula.particles.particle_data_builder.ParticleDataBuilder._distribution_type

*attribute* · *instance attribute*

```python
_distribution_type: str | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L68-L68)

### particula.particles.particle_data_builder.ParticleDataBuilder._masses

*attribute* · *instance attribute*

```python
_masses: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L60-L60)

### particula.particles.particle_data_builder.ParticleDataBuilder._n_boxes

*attribute* · *instance attribute*

```python
_n_boxes: Optional[int] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L65-L65)

### particula.particles.particle_data_builder.ParticleDataBuilder._n_particles

*attribute* · *instance attribute*

```python
_n_particles: Optional[int] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L66-L66)

### particula.particles.particle_data_builder.ParticleDataBuilder._n_species

*attribute* · *instance attribute*

```python
_n_species: Optional[int] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L67-L67)

### particula.particles.particle_data_builder.ParticleDataBuilder._radius_grid

*attribute* · *instance attribute*

```python
_radius_grid: NDArray[np.float64] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L69-L69)

### particula.particles.particle_data_builder.ParticleDataBuilder._volume

*attribute* · *instance attribute*

```python
_volume: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L64-L64)

### particula.particles.particle_data_builder.ParticleDataBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize empty builder state.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L58-L70)

### particula.particles.particle_data_builder.ParticleDataBuilder._broadcast_if_needed

*method*

```python
def _broadcast_if_needed(array: NDArray[np.float64], target_shape: tuple[int, ...], label: str) -> NDArray[np.float64]
```

Broadcast array to target shape when compatible.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L267-L289)

### particula.particles.particle_data_builder.ParticleDataBuilder._ensure_counts

*method*

```python
def _ensure_counts() -> None
```

Ensure counts are available for zero-init path.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L291-L300)

### particula.particles.particle_data_builder.ParticleDataBuilder._get_counts

*method*

```python
def _get_counts() -> tuple[int, int, int]
```

Return the integer counts currently tracked.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L302-L310)

### particula.particles.particle_data_builder.ParticleDataBuilder._infer_counts_from_masses

*method*

```python
def _infer_counts_from_masses() -> tuple[int, int, int]
```

Derive counts from masses when present.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L257-L265)

### particula.particles.particle_data_builder.ParticleDataBuilder.build

*method*

```python
def build() -> ParticleData
```

Construct a ``ParticleData`` instance with validation.

Require an explicit kind and matching concentration units: m^-3
for discrete bins, m^-4 for PDF nodes, or counts for resolved slots.
Validate the kind-specific volume and PDF radius grid before return.

**Returns**

- (`ParticleData`) — A validated, explicitly tagged ``ParticleData`` object.

**Raises**

- `ValueError` — When required fields, units, shapes, or physical
values do not match the declared representation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L312-L407)

### particula.particles.particle_data_builder.ParticleDataBuilder.set_charge

*method*

```python
def set_charge(charge: NDArray[np.float64]) -> 'ParticleDataBuilder'
```

Set particle charge (unitless) with batch handling.

**Parameters**

- `charge` (`NDArray[np.float64]`) — Charge array shaped (n_boxes, n_particles) or (n_particles,).

**Returns**

- (`'ParticleDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When charge is not 1D or 2D.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L168-L188)

### particula.particles.particle_data_builder.ParticleDataBuilder.set_concentration

*method*

```python
def set_concentration(concentration: NDArray[np.float64], units: str = '1/m^3') -> 'ParticleDataBuilder'
```

Set number concentration with optional unit conversion.

**Parameters**

- `concentration` (`NDArray[np.float64]`) — Concentration array shaped (n_boxes, n_particles) or (n_particles,).
- `units` (`str`) (default: `'1/m^3'`) — Storage units. PMF: 1/m^3 or 1/cm^3; PDF: 1/m^4; resolved: count.

**Returns**

- (`'ParticleDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When concentration is not 1D or 2D.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L136-L166)

### particula.particles.particle_data_builder.ParticleDataBuilder.set_density

*method*

```python
def set_density(density: NDArray[np.float64], units: str = 'kg/m^3') -> 'ParticleDataBuilder'
```

Set species densities with optional unit conversion.

**Parameters**

- `density` (`NDArray[np.float64]`) — Density array shaped (n_species,).
- `units` (`str`) (default: `'kg/m^3'`) — Units of the provided density. Supported: kg/m^3, g/cm^3.

**Returns**

- (`'ParticleDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When density is not 1D.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L190-L213)

### particula.particles.particle_data_builder.ParticleDataBuilder.set_distribution_type

*method*

```python
def set_distribution_type(kind: str) -> 'ParticleDataBuilder'
```

Declare the discrete, continuous PDF, or resolved storage kind.

**Parameters**

- `kind` (`str`) — Explicit ``discrete``, ``continuous_pdf``, or ``particle_resolved`` interpretation.

**Returns**

- (`'ParticleDataBuilder'`) — This builder for fluent chaining.

**Raises**

- `ValueError` — If the kind is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L72-L88)

### particula.particles.particle_data_builder.ParticleDataBuilder.set_masses

*method*

```python
def set_masses(masses: NDArray[np.float64], units: str = 'kg') -> 'ParticleDataBuilder'
```

Set per-species masses with optional unit conversion.

**Parameters**

- `masses` (`NDArray[np.float64]`) — Mass array shaped (n_boxes, n_particles, n_species) or (n_particles, n_species).
- `units` (`str`) (default: `'kg'`) — Units of the provided masses. Supported: kg, g, ug, ng.

**Returns**

- (`'ParticleDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When masses are not 2D or 3D.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L108-L134)

### particula.particles.particle_data_builder.ParticleDataBuilder.set_n_boxes

*method*

```python
def set_n_boxes(n_boxes: int) -> 'ParticleDataBuilder'
```

Set number of boxes for zero-init or shape validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L242-L245)

### particula.particles.particle_data_builder.ParticleDataBuilder.set_n_particles

*method*

```python
def set_n_particles(n_particles: int) -> 'ParticleDataBuilder'
```

Set number of particles for zero-init or shape validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L247-L250)

### particula.particles.particle_data_builder.ParticleDataBuilder.set_n_species

*method*

```python
def set_n_species(n_species: int) -> 'ParticleDataBuilder'
```

Set number of species for zero-init or shape validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L252-L255)

### particula.particles.particle_data_builder.ParticleDataBuilder.set_radius_grid

*method*

```python
def set_radius_grid(radius_grid: NDArray[np.float64]) -> 'ParticleDataBuilder'
```

Copy the explicit continuous-PDF radius grid in metres.

**Parameters**

- `radius_grid` (`NDArray[np.float64]`) — Radius nodes with shape (n_particles,).

**Returns**

- (`'ParticleDataBuilder'`) — This builder for fluent chaining.

**Raises**

- `ValueError` — At build time if the grid is invalid or the
declared kind is not ``continuous_pdf``.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L90-L106)

### particula.particles.particle_data_builder.ParticleDataBuilder.set_volume

*method*

```python
def set_volume(volume: NDArray[np.float64], units: str = 'm^3') -> 'ParticleDataBuilder'
```

Set per-box volume with optional unit conversion.

**Parameters**

- `volume` (`NDArray[np.float64]`) — Volume scalar or array shaped (n_boxes,).
- `units` (`str`) (default: `'m^3'`) — Units of the provided volume. Supported: m^3, cm^3, L.

**Returns**

- (`'ParticleDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When volume is neither scalar nor 1D.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_builder.py#L215-L240)
