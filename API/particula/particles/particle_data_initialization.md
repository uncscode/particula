# particula.particles.particle_data_initialization

*module*

Native particle initialization without strategy-bearing facades.

Initialization allocates owned float64 arrays. Aggregate wrapping is a separate,
identity-preserving operation. Binned densities use fixed volume 1 m^3.
Canonical SI inputs do not require Pint; other unit spellings/conversions do.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L1-L1)

## particula.particles.particle_data_initialization._bin_values

*function*

```python
def _bin_values(value: ArrayLike, count: int, name: str) -> NDArray[np.float64]
```

Broadcast scalar slot values without discarding any supplied lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L47-L54)

## particula.particles.particle_data_initialization._modal_parameters

*function*

```python
def _modal_parameters(mode: ArrayLike | None, geometric_standard_deviation: ArrayLike | None, number_concentration: ArrayLike | None, radius_units: str, number_concentration_units: str) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Normalize existing modal defaults and validate generator inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L57-L93)

## particula.particles.particle_data_initialization._reference_volume

*function*

```python
def _reference_volume(value: float, units: str) -> float
```

Canonicalize conversion roundoff only for non-SI reference volumes.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L96-L101)

## particula.particles.particle_data_initialization._single_density

*function*

```python
def _single_density(value: ArrayLike, units: str) -> float
```

Require a single material; redundant equal density values are allowed.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L36-L44)

## particula.particles.particle_data_initialization._slot_values

*function*

```python
def _slot_values(value: ArrayLike, shape: tuple[int, int, int], name: str, *, reduce_species: bool = False) -> NDArray[np.float64]
```

Copy scalar/slot values; reduce redundant charge species lanes only.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L281-L304)

## particula.particles.particle_data_initialization._species_density

*function*

```python
def _species_density(value: ArrayLike, shape: tuple[int, int, int]) -> NDArray[np.float64]
```

Reduce only scalar, species or redundant population density forms.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L260-L278)

## particula.particles.particle_data_initialization._unit_factor

*function*

```python
def _unit_factor(units: str, target: str) -> float
```

Avoid the optional converter when inputs already use canonical units.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L21-L23)

## particula.particles.particle_data_initialization._vector

*function*

```python
def _vector(value: ArrayLike, name: str) -> NDArray[np.float64]
```

Copy a scalar, vector, or singleton row/column into one dimension.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L26-L33)

## particula.particles.particle_data_initialization.create_lognormal_binned_particle_data

*function*

```python
def create_lognormal_binned_particle_data(distribution_type: str, radius_bins: ArrayLike | None = None, mode: ArrayLike | None = None, geometric_standard_deviation: ArrayLike | None = None, number_concentration: ArrayLike | None = None, density: ArrayLike = 1000.0, charge: ArrayLike = 0.0, volume: float = 1.0, radius_units: str = 'm', number_concentration_units: str = '1/m^3', density_units: str = 'kg/m^3', volume_units: str = 'm^3') -> ParticleData
```

Initialize native bins with the existing lognormal PDF/PMF generators.

**Parameters**

- `distribution_type` (`str`) — Required ``discrete`` or ``continuous_pdf``.
- `radius_bins` (`ArrayLike | None`) (default: `None`) — Radius nodes; default logspace(-9, -4, 250) metres.
- `mode` (`ArrayLike | None`) (default: `None`) — Lognormal scale radii, default [100e-9, 1e-6] metres.
- `geometric_standard_deviation` (`ArrayLike | None`) (default: `None`) — GSD values > 1, default [1.2, 1.4].
- `number_concentration` (`ArrayLike | None`) (default: `None`) — Modal number densities, default [1e10, 1e9] m^-3.
- `density` (`ArrayLike`) (default: `1000.0`) — Single material density, default 1000 kg/m^3.
- `charge` (`ArrayLike`) (default: `0.0`) — Scalar or per-node charge, default zero.
- `volume` (`float`) (default: `1.0`) — Reference volume; must convert to exactly 1 m^3.
- `radius_units` (`str`) (default: `'m'`) — Units of supplied nodes/modes; defaults remain SI.
- `number_concentration_units` (`str`) (default: `'1/m^3'`) — Units of supplied modal densities.
- `density_units` (`str`) (default: `'kg/m^3'`) — Material-density units.
- `volume_units` (`str`) (default: `'m^3'`) — Reference-volume units.

**Returns**

- (`ParticleData`) — Owned single-box ParticleData. The existing generator normalizes modes
- (`ParticleData`) — on the supplied radius interval; PDF density is integrated over radius,
- (`ParticleData`) — whereas discrete bin densities are summed. No facade is constructed.

**Raises**

- `ValueError` — If inputs do not define a valid distribution. Discrete PMFs
require at least three positive, strictly increasing nodes; PDFs
require at least two. The existing PMF converter needs two
intervals to extrapolate its last bin width.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L171-L257)

## particula.particles.particle_data_initialization.create_lognormal_resolved_particle_data

*function*

```python
def create_lognormal_resolved_particle_data(particle_resolved_count: int = 20000, mode: ArrayLike | None = None, geometric_standard_deviation: ArrayLike | None = None, number_concentration: ArrayLike | None = None, density: ArrayLike = 1000.0, charge: ArrayLike | None = 0.0, volume: float = 1.0, radius_units: str = 'm', number_concentration_units: str = '1/m^3', density_units: str = 'kg/m^3', volume_units: str = 'm^3') -> ParticleData
```

Sample the existing lognormal mixture with one count per sampled slot.

**Parameters**

- `particle_resolved_count` (`int`) (default: `20000`) — Positive integer sample count, default 20,000.
- `mode` (`ArrayLike | None`) (default: `None`) — Scale radii; default [100e-9, 1e-6] metres.
- `geometric_standard_deviation` (`ArrayLike | None`) (default: `None`) — GSD values > 1; default [1.2, 1.4].
- `number_concentration` (`ArrayLike | None`) (default: `None`) — Modal densities; default [1e10, 1e9] m^-3. These set relative modal sampling weights, not final number density. Their sum must be finite before sampling.
- `density` (`ArrayLike`) (default: `1000.0`) — Single material density, default 1000 kg/m^3.
- `charge` (`ArrayLike | None`) (default: `0.0`) — Scalar or per-particle charge; default zero.
- `volume` (`float`) (default: `1.0`) — Positive finite physical volume, default 1 m^3.
- `radius_units` (`str`) (default: `'m'`) — Units of supplied modes; omitted defaults are SI.
- `number_concentration_units` (`str`) (default: `'1/m^3'`) — Units of supplied modal densities.
- `density_units` (`str`) (default: `'kg/m^3'`) — Material-density units.
- `volume_units` (`str`) (default: `'m^3'`) — Physical-volume units.

**Returns**

- (`ParticleData`) — Owned single-box data with unit count weights, hence number density
- (`ParticleData`) — equal to sample count divided by physical volume. The existing sampler
- (`ParticleData`) — and NumPy/SciPy global random-state behavior are preserved; no new seed
- (`ParticleData`) — policy or cross-backend replay guarantee is introduced.

**Raises**

- `ValueError` — If parameters, count, shape or physical values are invalid.
Deterministic input admission precedes random sampling.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L402-L477)

## particula.particles.particle_data_initialization.create_particle_data_from_radii

*function*

```python
def create_particle_data_from_radii(radii: ArrayLike, concentration: ArrayLike, distribution_type: str, density: ArrayLike = 1000.0, charge: ArrayLike = 0.0, volume: float = 1.0, radius_units: str = 'm', concentration_units: str | None = None, density_units: str = 'kg/m^3', volume_units: str = 'm^3') -> ParticleData
```

Create owned single-box, single-material data from particle radii.

**Parameters**

- `radii` (`ArrayLike`) — Scalar, vector or singleton row/column of positive radii.
- `concentration` (`ArrayLike`) — Scalar or matching per-radius densities.
- `distribution_type` (`str`) — Required ``discrete`` or ``continuous_pdf`` kind.
- `density` (`ArrayLike`) (default: `1000.0`) — One positive material density; redundant equal values allowed.
- `charge` (`ArrayLike`) (default: `0.0`) — Scalar or per-radius elementary-charge counts, default zero.
- `volume` (`float`) (default: `1.0`) — Reference volume, which must convert to exactly 1 m^3.
- `radius_units` (`str`) (default: `'m'`) — Radius units, default metres.
- `concentration_units` (`str | None`) (default: `None`) — Defaults to 1/m^3 for bins, 1/m^4 for PDFs. Discrete inputs also accept 1/cm^3, as in ParticleDataBuilder.
- `density_units` (`str`) (default: `'kg/m^3'`) — Material-density units, default kg/m^3.
- `volume_units` (`str`) (default: `'m^3'`) — Reference-volume units, default m^3. Non-SI conversion roundoff within four float64 ulps of one is canonicalized to one.

**Returns**

- (`ParticleData`) — Owned float64 data with masses (1, N, 1), density (1,), and volume
- (`ParticleData`) — (1,). PDF data retains its explicit radius grid. Mass is 4*pi*r^3*rho/3.

**Raises**

- `ValueError` — If kind, units, shapes, physical inputs or PDF grid fail
admission. No caller arrays are modified. Empty discrete input is
supported; PDF grids require at least two increasing nodes.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L104-L168)

## particula.particles.particle_data_initialization.create_resolved_particle_data

*function*

```python
def create_resolved_particle_data(masses: ArrayLike, density: ArrayLike, concentration: ArrayLike | None = None, charge: ArrayLike | None = 0.0, volume: ArrayLike = 1.0, mass_units: str = 'kg', density_units: str = 'kg/m^3', concentration_units: str = 'count', volume_units: str = 'm^3') -> ParticleData
```

Create owned resolved data with explicit count or density input units.

**Parameters**

- `masses` (`ArrayLike`) — Scalar, (N,) single-species, (N,S), or (B,N,S) masses.
- `density` (`ArrayLike`) — Scalar or (S,) densities; singleton forms and identical population rows are accepted. Differing population rows reject.
- `concentration` (`ArrayLike | None`) (default: `None`) — Scalar, (N,), or (B,N) weights; default one count/slot. Singleton row/column forms broadcast without dropping values.
- `charge` (`ArrayLike | None`) (default: `0.0`) — Scalar, (N,), or (B,N) charge counts; None means zero. Redundant (N,S) or (B,N,S) species lanes may reduce only when equal. Canonical (B,N) shape takes precedence over a redundant shape.
- `volume` (`ArrayLike`) (default: `1.0`) — Positive finite physical volume, scalar or (B,).
- `mass_units` (`str`) (default: `'kg'`) — Per-species mass units, default kg.
- `density_units` (`str`) (default: `'kg/m^3'`) — Material-density units, default kg/m^3.
- `concentration_units` (`str`) (default: `'count'`) — ``count`` preserves weights; ``1/m^3`` or ``1/cm^3`` explicitly converts density to counts using volume once. Omitted concentration requires ``count`` units.
- `volume_units` (`str`) (default: `'m^3'`) — Physical volume units, default m^3.

**Returns**

- (`ParticleData`) — Owned float64 ParticleData, explicitly particle_resolved. Empty slot
- (`ParticleData`) — axes and fractional/zero weights are supported. This does not authorize
- (`ParticleData`) — scientific algorithms that require unit weights or a single box.

**Raises**

- `ValueError` — If units, shapes or representation values are invalid.
No caller arrays are changed and no species/particle lane is lost.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/particle_data_initialization.py#L307-L399)
