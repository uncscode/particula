# particula.optics.cross_sections

*module*

Dimensional and normalized optical quantities for individual spheres.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L1-L1)

## particula.optics.cross_sections.DifferentialCrossSections

*class*

```python
class DifferentialCrossSections(NamedTuple)
```

Bases: `NamedTuple`

Angular scattering areas per solid angle in m²/sr.

**Attributes**

- `perpendicular` (`NDArray[np.float64]`) — Perpendicular incident polarization, |S1|²/k².
- `parallel` (`NDArray[np.float64]`) — Parallel incident polarization, |S2|²/k².
- `unpolarized` (`NDArray[np.float64]`) — Mean of the two polarized cross sections.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L136-L147)

### particula.optics.cross_sections.DifferentialCrossSections.parallel

*attribute* · *instance attribute*

```python
parallel: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L146-L146)

### particula.optics.cross_sections.DifferentialCrossSections.perpendicular

*attribute* · *instance attribute*

```python
perpendicular: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L145-L145)

### particula.optics.cross_sections.DifferentialCrossSections.unpolarized

*attribute* · *instance attribute*

```python
unpolarized: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L147-L147)

## particula.optics.cross_sections.OpticalCrossSections

*class*

```python
class OpticalCrossSections(NamedTuple)
```

Bases: `NamedTuple`

Per-particle optical areas in m², with the broadcast input shape.

**Attributes**

- `extinction` (`NDArray[np.float64]`) — Extinction cross section [m²].
- `scattering` (`NDArray[np.float64]`) — Scattering cross section [m²].
- `absorption` (`NDArray[np.float64]`) — Absorption cross section [m²].
- `backscatter` (`NDArray[np.float64]`) — Radar backscatter cross section [m²], not m²/sr.
- `radiation_pressure` (`NDArray[np.float64]`) — Radiation-pressure cross section [m²].

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L11-L26)

### particula.optics.cross_sections.OpticalCrossSections.absorption

*attribute* · *instance attribute*

```python
absorption: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L24-L24)

### particula.optics.cross_sections.OpticalCrossSections.backscatter

*attribute* · *instance attribute*

```python
backscatter: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L25-L25)

### particula.optics.cross_sections.OpticalCrossSections.extinction

*attribute* · *instance attribute*

```python
extinction: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L22-L22)

### particula.optics.cross_sections.OpticalCrossSections.radiation_pressure

*attribute* · *instance attribute*

```python
radiation_pressure: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L26-L26)

### particula.optics.cross_sections.OpticalCrossSections.scattering

*attribute* · *instance attribute*

```python
scattering: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L23-L23)

## particula.optics.cross_sections.get_cross_sections

*function*

```python
def get_cross_sections(efficiencies: MieEfficiencies, radius: ArrayLike) -> OpticalCrossSections
```

Multiply efficiencies by geometric area pi*r² without recalculating.

**Parameters**

- `efficiencies` (`MieEfficiencies`) — Output of get_mie_efficiencies for the same radius.
- `radius` (`ArrayLike`) — Nonnegative finite radius [m], broadcastable with efficiencies.

**Returns**

- (`OpticalCrossSections`) — Named cross sections [m²]. Inputs are not mutated.

**Raises**

- `ValueError` — If the radius is negative or nonfinite.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L29-L59)

## particula.optics.cross_sections.get_differential_cross_sections

*function*

```python
def get_differential_cross_sections(perpendicular: ArrayLike, parallel: ArrayLike, wavelength: ArrayLike, medium_index: ArrayLike = 1.0) -> DifferentialCrossSections
```

Convert S1/S2 amplitudes to polarized differential cross sections.

Uses the nonabsorbing-medium wave number k=2*pi*n_medium/lambda_vacuum,
following Bohren and Huffman (1983), chapter 4,
https://doi.org/10.1002/9783527618156.

**Parameters**

- `perpendicular` (`ArrayLike`) — Dimensionless S1, shaped batch+(angles,).
- `parallel` (`ArrayLike`) — Dimensionless S2, same shape as perpendicular.
- `wavelength` (`ArrayLike`) — Positive finite vacuum wavelength [m]; broadcasts with batch axes, not the angle axis.
- `medium_index` (`ArrayLike`) (default: `1.0`) — Positive finite real medium index, default one; broadcasts with batch axes.

**Returns**

- (`DifferentialCrossSections`) — Perpendicular, parallel and unpolarized differential scattering
- (`DifferentialCrossSections`) — cross sections [m²/sr], shaped broadcast batch+(angles,).
- (`DifferentialCrossSections`) — Inputs are unchanged; scalar angles must retain a length-one axis.

**Raises**

- `ValueError` — If amplitudes, shapes or physical inputs are invalid.
- `RuntimeError` — If dimensional cross sections overflow.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L150-L205)

## particula.optics.cross_sections.get_differential_mueller_matrix

*function*

```python
def get_differential_mueller_matrix(matrix: ArrayLike, wavelength: ArrayLike, medium_index: ArrayLike = 1.0) -> NDArray[np.float64]
```

Convert unnormalized angular Mueller values to m²/sr.

**Parameters**

- `matrix` (`ArrayLike`) — Raw Mueller matrix shaped batch+(angles,4,4).
- `wavelength` (`ArrayLike`) — Positive vacuum wavelength [m], broadcastable with batch.
- `medium_index` (`ArrayLike`) (default: `1.0`) — Positive real medium index, broadcastable with batch.

**Returns**

- (`NDArray[np.float64]`) — Matrix divided by (2*pi*n_medium/lambda_vacuum)² [m²/sr].

**Raises**

- `ValueError` — If shapes or physical values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L62-L97)

## particula.optics.cross_sections.get_phase_matrix

*function*

```python
def get_phase_matrix(differential_matrix: ArrayLike, scattering_cross_section: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]
```

Normalize a differential Mueller matrix by total scattering.

**Parameters**

- `differential_matrix` (`ArrayLike`) — Angular matrix [m²/sr], batch+(angles,4,4).
- `scattering_cross_section` (`ArrayLike`) — Nonnegative total scattering [m²], broadcastable with batch, evaluated for the same particle.

**Returns**

- (`NDArray[np.float64]`) — Phase matrix [sr⁻¹] and batch validity mask. Element 11 integrates
- (`NDArray[np.bool_]`) — to one over solid angle for valid cases. Zero scattering returns a
- (`tuple[NDArray[np.float64], NDArray[np.bool_]]`) — zero matrix and False; this is not a uniform-scattering assumption.

**Raises**

- `ValueError` — If shapes or physical values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L100-L133)
