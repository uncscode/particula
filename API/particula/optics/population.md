# particula.optics.population

*module*

Number-weighted optical properties of independent spherical particles.

Population sums assume incoherent, independent scattering. Number weights
are densities [m⁻³], including quadrature weights for continuous radius PDFs.
Bulk asymmetry is weighted by scattering, not by particle number alone.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L1-L1)

## particula.optics.population.PopulationOptics

*class*

```python
class PopulationOptics(NamedTuple)
```

Bases: `NamedTuple`

Bulk optical coefficients and intensive properties.

**Attributes**

- `extinction` (`NDArray[np.float64]`) — Extinction coefficient [m⁻¹].
- `scattering` (`NDArray[np.float64]`) — Scattering coefficient [m⁻¹].
- `absorption` (`NDArray[np.float64]`) — Absorption coefficient [m⁻¹].
- `backscatter` (`NDArray[np.float64]`) — Radar backscatter coefficient [m⁻¹].
- `radiation_pressure` (`NDArray[np.float64]`) — Radiation-pressure coefficient [m⁻¹].
- `single_scattering_albedo` (`NDArray[np.float64]`) — Scattering/extinction, zero if undefined.
- `asymmetry` (`NDArray[np.float64]`) — Scattering-weighted cosine, zero if undefined.
- `extinction_valid` (`NDArray[np.bool_]`) — Whether single-scattering albedo is defined.
- `scattering_valid` (`NDArray[np.bool_]`) — Whether asymmetry is defined.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L16-L39)

### particula.optics.population.PopulationOptics.absorption

*attribute* · *instance attribute*

```python
absorption: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L33-L33)

### particula.optics.population.PopulationOptics.asymmetry

*attribute* · *instance attribute*

```python
asymmetry: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L37-L37)

### particula.optics.population.PopulationOptics.backscatter

*attribute* · *instance attribute*

```python
backscatter: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L34-L34)

### particula.optics.population.PopulationOptics.extinction

*attribute* · *instance attribute*

```python
extinction: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L31-L31)

### particula.optics.population.PopulationOptics.extinction_valid

*attribute* · *instance attribute*

```python
extinction_valid: NDArray[np.bool_]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L38-L38)

### particula.optics.population.PopulationOptics.radiation_pressure

*attribute* · *instance attribute*

```python
radiation_pressure: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L35-L35)

### particula.optics.population.PopulationOptics.scattering

*attribute* · *instance attribute*

```python
scattering: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L32-L32)

### particula.optics.population.PopulationOptics.scattering_valid

*attribute* · *instance attribute*

```python
scattering_valid: NDArray[np.bool_]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L39-L39)

### particula.optics.population.PopulationOptics.single_scattering_albedo

*attribute* · *instance attribute*

```python
single_scattering_albedo: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L36-L36)

## particula.optics.population._nonnegative

*function*

```python
def _nonnegative(values: ArrayLike, name: str) -> NDArray[np.float64]
```

Coerce a finite nonnegative physical array without modifying it.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L42-L49)

## particula.optics.population.get_population_mueller_matrix

*function*

```python
def get_population_mueller_matrix(differential_matrix: ArrayLike, number_density_weights: ArrayLike, *, particle_axis: int = -1) -> NDArray[np.float64]
```

Sum independent-particle differential Mueller matrices by number.

**Parameters**

- `differential_matrix` (`ArrayLike`) — Per-particle matrix [m²/sr] with shape batch+(angles,4,4), where batch includes the particle axis. All matrices must use the same incident/outgoing reference frames.
- `number_density_weights` (`ArrayLike`) — Nonnegative weights [m⁻³] broadcastable with batch. Negative polarized matrix entries are valid.
- `particle_axis` (`int`) (default: `-1`) — Particle axis within the broadcast batch shape.

**Returns**

- (`NDArray[np.float64]`) — Bulk differential matrix [m⁻¹/sr]. Normalize by bulk scattering
- (`NDArray[np.float64]`) — coefficient to obtain the population phase matrix.

**Raises**

- `ValueError` — If inputs or resulting matrix values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L189-L231)

## particula.optics.population.get_population_optics

*function*

```python
def get_population_optics(cross_sections: OpticalCrossSections, asymmetry: ArrayLike, number_density_weights: ArrayLike, *, particle_axis: int = -1) -> PopulationOptics
```

Reduce per-particle optical areas to bulk optical coefficients.

**Parameters**

- `cross_sections` (`OpticalCrossSections`) — Nonnegative per-particle areas [m²].
- `asymmetry` (`ArrayLike`) — Per-particle scattering-weighted cosine in [-1,1].
- `number_density_weights` (`ArrayLike`) — Number weights [m⁻³]; use explicit singleton axes for wavelength broadcasting, e.g. (boxes,1,particles).
- `particle_axis` (`int`) (default: `-1`) — Particle axis in the broadcast shape, default last.

**Returns**

- (`PopulationOptics`) — Bulk coefficients [m⁻¹], albedo, asymmetry and validity masks with
- (`PopulationOptics`) — the particle axis removed. Empty/zero populations produce zeros.

**Raises**

- `ValueError` — If inputs, reduction axis or resulting values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L117-L186)

## particula.optics.population.get_population_weights

*function*

```python
def get_population_weights(concentration: ArrayLike, distribution_type: str, *, volume: ArrayLike | None = None, radius_grid: ArrayLike | None = None) -> NDArray[np.float64]
```

Convert native population measures to number-density weights.

**Parameters**

- `concentration` (`ArrayLike`) — (..., particles) number density [m⁻³] for discrete, represented count for particle_resolved, or dn/dr [m⁻⁴] for continuous_pdf. The final axis is always the particle axis.
- `distribution_type` (`str`) — discrete, particle_resolved, or continuous_pdf.
- `volume` (`ArrayLike | None`) (default: `None`) — Required positive physical volume [m³] for resolved counts, broadcastable with concentration's leading axes. Omit otherwise.
- `radius_grid` (`ArrayLike | None`) (default: `None`) — Required positive increasing 1D radius nodes [m] for continuous_pdf. At least two nodes; omit for other measures.

**Returns**

- (`NDArray[np.float64]`) — New number-density weights [m⁻³]. PDF weights implement trapezoidal
- (`NDArray[np.float64]`) — integration of the complete optical-cross-section times PDF product.

**Raises**

- `ValueError` — If the measure, shape, values or supplied metadata conflict.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L52-L114)
