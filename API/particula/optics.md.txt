# particula.optics

*module*

Optical functions for spherical aerosol particles in SI units.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/__init__.py#L1-L1)

## particula.optics.ColumnOptics

*class*

```python
class ColumnOptics(NamedTuple)
```

Bases: `NamedTuple`

Re-exported from: `particula.optics.column`

Layer and integrated optical depth and direct-beam transmission.

**Attributes**

- `layer_optical_depth` (`NDArray[np.float64]`) — Extinction times geometric path length [1].
- `optical_depth` (`NDArray[np.float64]`) — Sum of layer depths along the selected layer axis [1].
- `transmission` (`NDArray[np.float64]`) — Unscattered direct-beam fraction exp(-optical_depth).

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L15-L26)

### particula.optics.ColumnOptics.layer_optical_depth

*attribute* · *instance attribute*

```python
layer_optical_depth: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L24-L24)

### particula.optics.ColumnOptics.optical_depth

*attribute* · *instance attribute*

```python
optical_depth: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L25-L25)

### particula.optics.ColumnOptics.transmission

*attribute* · *instance attribute*

```python
transmission: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L26-L26)

## particula.optics.DifferentialCrossSections

*class*

```python
class DifferentialCrossSections(NamedTuple)
```

Bases: `NamedTuple`

Re-exported from: `particula.optics.cross_sections`

Angular scattering areas per solid angle in m²/sr.

**Attributes**

- `perpendicular` (`NDArray[np.float64]`) — Perpendicular incident polarization, |S1|²/k².
- `parallel` (`NDArray[np.float64]`) — Parallel incident polarization, |S2|²/k².
- `unpolarized` (`NDArray[np.float64]`) — Mean of the two polarized cross sections.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L136-L147)

### particula.optics.DifferentialCrossSections.parallel

*attribute* · *instance attribute*

```python
parallel: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L146-L146)

### particula.optics.DifferentialCrossSections.perpendicular

*attribute* · *instance attribute*

```python
perpendicular: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L145-L145)

### particula.optics.DifferentialCrossSections.unpolarized

*attribute* · *instance attribute*

```python
unpolarized: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L147-L147)

## particula.optics.MieCoefficients

*class*

```python
class MieCoefficients(NamedTuple)
```

Bases: `NamedTuple`

Re-exported from: `particula.optics.mie`

Complex coefficients, zero padded along the final order axis.

**Attributes**

- `electric` (`NDArray[np.complex128]`) — Electric coefficients a_n, starting at order one.
- `magnetic` (`NDArray[np.complex128]`) — Magnetic coefficients b_n, starting at order one.
- `term_count` (`NDArray[np.int64]`) — Valid number of coefficients for each broadcast case.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L15-L26)

### particula.optics.MieCoefficients.electric

*attribute* · *instance attribute*

```python
electric: NDArray[np.complex128]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L24-L24)

### particula.optics.MieCoefficients.magnetic

*attribute* · *instance attribute*

```python
magnetic: NDArray[np.complex128]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L25-L25)

### particula.optics.MieCoefficients.term_count

*attribute* · *instance attribute*

```python
term_count: NDArray[np.int64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L26-L26)

## particula.optics.MieEfficiencies

*class*

```python
class MieEfficiencies(NamedTuple)
```

Bases: `NamedTuple`

Re-exported from: `particula.optics.mie`

Dimensionless efficiencies and asymmetry for broadcast input cases.

**Attributes**

- `extinction` (`NDArray[np.float64]`) — Total extinction efficiency.
- `scattering` (`NDArray[np.float64]`) — Scattering efficiency.
- `absorption` (`NDArray[np.float64]`) — Absorption efficiency.
- `asymmetry` (`NDArray[np.float64]`) — Scattering-weighted cosine, zero when undefined.
- `backscatter` (`NDArray[np.float64]`) — Radar-convention backscatter efficiency.
- `radiation_pressure` (`NDArray[np.float64]`) — Extinction minus asymmetry times scattering.
- `scattering_valid` (`NDArray[np.bool_]`) — Whether asymmetry is defined.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L29-L48)

### particula.optics.MieEfficiencies.absorption

*attribute* · *instance attribute*

```python
absorption: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L44-L44)

### particula.optics.MieEfficiencies.asymmetry

*attribute* · *instance attribute*

```python
asymmetry: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L45-L45)

### particula.optics.MieEfficiencies.backscatter

*attribute* · *instance attribute*

```python
backscatter: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L46-L46)

### particula.optics.MieEfficiencies.extinction

*attribute* · *instance attribute*

```python
extinction: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L42-L42)

### particula.optics.MieEfficiencies.radiation_pressure

*attribute* · *instance attribute*

```python
radiation_pressure: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L47-L47)

### particula.optics.MieEfficiencies.scattering

*attribute* · *instance attribute*

```python
scattering: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L43-L43)

### particula.optics.MieEfficiencies.scattering_valid

*attribute* · *instance attribute*

```python
scattering_valid: NDArray[np.bool_]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L48-L48)

## particula.optics.OpticalCrossSections

*class*

```python
class OpticalCrossSections(NamedTuple)
```

Bases: `NamedTuple`

Re-exported from: `particula.optics.cross_sections`

Per-particle optical areas in m², with the broadcast input shape.

**Attributes**

- `extinction` (`NDArray[np.float64]`) — Extinction cross section [m²].
- `scattering` (`NDArray[np.float64]`) — Scattering cross section [m²].
- `absorption` (`NDArray[np.float64]`) — Absorption cross section [m²].
- `backscatter` (`NDArray[np.float64]`) — Radar backscatter cross section [m²], not m²/sr.
- `radiation_pressure` (`NDArray[np.float64]`) — Radiation-pressure cross section [m²].

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L11-L26)

### particula.optics.OpticalCrossSections.absorption

*attribute* · *instance attribute*

```python
absorption: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L24-L24)

### particula.optics.OpticalCrossSections.backscatter

*attribute* · *instance attribute*

```python
backscatter: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L25-L25)

### particula.optics.OpticalCrossSections.extinction

*attribute* · *instance attribute*

```python
extinction: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L22-L22)

### particula.optics.OpticalCrossSections.radiation_pressure

*attribute* · *instance attribute*

```python
radiation_pressure: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L26-L26)

### particula.optics.OpticalCrossSections.scattering

*attribute* · *instance attribute*

```python
scattering: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L23-L23)

## particula.optics.PopulationOptics

*class*

```python
class PopulationOptics(NamedTuple)
```

Bases: `NamedTuple`

Re-exported from: `particula.optics.population`

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

### particula.optics.PopulationOptics.absorption

*attribute* · *instance attribute*

```python
absorption: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L33-L33)

### particula.optics.PopulationOptics.asymmetry

*attribute* · *instance attribute*

```python
asymmetry: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L37-L37)

### particula.optics.PopulationOptics.backscatter

*attribute* · *instance attribute*

```python
backscatter: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L34-L34)

### particula.optics.PopulationOptics.extinction

*attribute* · *instance attribute*

```python
extinction: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L31-L31)

### particula.optics.PopulationOptics.extinction_valid

*attribute* · *instance attribute*

```python
extinction_valid: NDArray[np.bool_]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L38-L38)

### particula.optics.PopulationOptics.radiation_pressure

*attribute* · *instance attribute*

```python
radiation_pressure: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L35-L35)

### particula.optics.PopulationOptics.scattering

*attribute* · *instance attribute*

```python
scattering: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L32-L32)

### particula.optics.PopulationOptics.scattering_valid

*attribute* · *instance attribute*

```python
scattering_valid: NDArray[np.bool_]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L39-L39)

### particula.optics.PopulationOptics.single_scattering_albedo

*attribute* · *instance attribute*

```python
single_scattering_albedo: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L36-L36)

## particula.optics.ScatteringAmplitudes

*class*

```python
class ScatteringAmplitudes(NamedTuple)
```

Bases: `NamedTuple`

Re-exported from: `particula.optics.angular`

Dimensionless complex amplitudes with a final angle axis.

**Attributes**

- `perpendicular` (`NDArray[np.complex128]`) — S1 amplitude perpendicular to the scattering plane.
- `parallel` (`NDArray[np.complex128]`) — S2 amplitude parallel to the scattering plane.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/angular.py#L16-L25)

### particula.optics.ScatteringAmplitudes.parallel

*attribute* · *instance attribute*

```python
parallel: NDArray[np.complex128]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/angular.py#L25-L25)

### particula.optics.ScatteringAmplitudes.perpendicular

*attribute* · *instance attribute*

```python
perpendicular: NDArray[np.complex128]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/angular.py#L24-L24)

## particula.optics.apply_mueller_matrix

*function*

```python
def apply_mueller_matrix(matrix: ArrayLike, stokes: ArrayLike, input_rotation: ArrayLike = 0.0, output_rotation: ArrayLike = 0.0) -> NDArray[np.float64]
```

Re-exported from: `particula.optics.polarization`

Apply R(output_rotation) M R(input_rotation) to a Stokes vector.

**Parameters**

- `matrix` (`ArrayLike`) — Finite (..., 4, 4) Mueller array; units set output units.
- `stokes` (`ArrayLike`) — Finite (..., 4) I,Q,U,V input, broadcastable with matrix.
- `input_rotation` (`ArrayLike`) (default: `0.0`) — Rotation [rad] into the matrix's incident basis.
- `output_rotation` (`ArrayLike`) (default: `0.0`) — Rotation [rad] from its outgoing basis.

**Returns**

- (`NDArray[np.float64]`) — New output Stokes vectors, without intensity normalization or
- (`NDArray[np.float64]`) — propagation-distance factors. Signed linear combinations are allowed.

**Raises**

- `ValueError` — If matrix/vector shapes or finite values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/polarization.py#L78-L106)

## particula.optics.get_column_optics

*function*

```python
def get_column_optics(extinction: ArrayLike, path_length: ArrayLike, *, layer_axis: int = -1) -> ColumnOptics
```

Re-exported from: `particula.optics.column`

Integrate extinction along explicit path segments.

**Parameters**

- `extinction` (`ArrayLike`) — Nonnegative extinction coefficient [m⁻¹].
- `path_length` (`ArrayLike`) — Nonnegative path through each layer [m], broadcastable with extinction. Supply slant lengths explicitly if applicable.
- `layer_axis` (`int`) (default: `-1`) — Layer axis after broadcasting; scalar input is one layer.

**Returns**

- (`ColumnOptics`) — Dimensionless layer depths, total depth and direct transmission.
- (`ColumnOptics`) — Empty columns have zero depth and unit transmission.

**Raises**

- `ValueError` — If inputs, resulting depths or layer axis are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L29-L69)

## particula.optics.get_cross_sections

*function*

```python
def get_cross_sections(efficiencies: MieEfficiencies, radius: ArrayLike) -> OpticalCrossSections
```

Re-exported from: `particula.optics.cross_sections`

Multiply efficiencies by geometric area pi*r² without recalculating.

**Parameters**

- `efficiencies` (`MieEfficiencies`) — Output of get_mie_efficiencies for the same radius.
- `radius` (`ArrayLike`) — Nonnegative finite radius [m], broadcastable with efficiencies.

**Returns**

- (`OpticalCrossSections`) — Named cross sections [m²]. Inputs are not mutated.

**Raises**

- `ValueError` — If the radius is negative or nonfinite.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/cross_sections.py#L29-L59)

## particula.optics.get_differential_cross_sections

*function*

```python
def get_differential_cross_sections(perpendicular: ArrayLike, parallel: ArrayLike, wavelength: ArrayLike, medium_index: ArrayLike = 1.0) -> DifferentialCrossSections
```

Re-exported from: `particula.optics.cross_sections`

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

## particula.optics.get_differential_mueller_matrix

*function*

```python
def get_differential_mueller_matrix(matrix: ArrayLike, wavelength: ArrayLike, medium_index: ArrayLike = 1.0) -> NDArray[np.float64]
```

Re-exported from: `particula.optics.cross_sections`

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

## particula.optics.get_mass_specific_opacity

*function*

```python
def get_mass_specific_opacity(coefficient: ArrayLike, mass_density: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]
```

Re-exported from: `particula.optics.column`

Divide an optical coefficient by a specified material mass density.

**Parameters**

- `coefficient` (`ArrayLike`) — Nonnegative extinction or absorption [m⁻¹].
- `mass_density` (`ArrayLike`) — Nonnegative density [kg/m³], broadcastable with coefficient. The caller specifies which material inventory it is.

**Returns**

- (`NDArray[np.float64]`) — Mass-specific opacity [m²/kg] and positive-density validity mask.
- (`NDArray[np.bool_]`) — A zero/zero vacuum returns zero and False.

**Raises**

- `ValueError` — If inputs/results are invalid or nonzero optics has zero
density.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L72-L104)

## particula.optics.get_mie_coefficients

*function*

```python
def get_mie_coefficients(relative_refractive_index: ArrayLike, size_parameter: ArrayLike) -> MieCoefficients
```

Re-exported from: `particula.optics.mie`

Calculate external coefficients for nonmagnetic homogeneous spheres.

**Parameters**

- `relative_refractive_index` (`ArrayLike`) — Particle/medium index n+ik, n>0, k>=0.
- `size_parameter` (`ArrayLike`) — Dimensionless 2*pi*r*n_medium/lambda_vacuum, 0..10000.

**Returns**

- (`MieCoefficients`) — Broadcast coefficients with a final, zero-padded order axis and
- (`MieCoefficients`) — per-case valid term counts. Inputs are not mutated.

**Raises**

- `ValueError` — If physical inputs are nonfinite or outside the domain.
- `RuntimeError` — If the numerical recurrence fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L120-L173)

## particula.optics.get_mie_efficiencies

*function*

```python
def get_mie_efficiencies(radius: ArrayLike, wavelength: ArrayLike, refractive_index: ArrayLike, medium_index: ArrayLike = 1.0) -> MieEfficiencies
```

Re-exported from: `particula.optics.mie`

Calculate efficiencies from SI radius and vacuum wavelength.

**Parameters**

- `radius` (`ArrayLike`) — Nonnegative sphere radius [m].
- `wavelength` (`ArrayLike`) — Positive vacuum wavelength [m].
- `refractive_index` (`ArrayLike`) — Absolute particle refractive index n+ik.
- `medium_index` (`ArrayLike`) (default: `1.0`) — Positive real nonabsorbing medium index, default 1.

**Returns**

- (`MieEfficiencies`) — Named dimensionless arrays with the broadcast input shape. Zero
- (`MieEfficiencies`) — scattering gives asymmetry zero and scattering_valid False.

**Raises**

- `ValueError` — If inputs violate the physical domain or x exceeds 10000.
- `RuntimeError` — If arithmetic is unresolved, coefficient calculation
fails or efficiencies violate energy/physical bounds beyond
64 float64 eps of roundoff. Negative absorption within that
roundoff is set to zero with extinction adjusted to scattering.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/mie.py#L176-L282)

## particula.optics.get_mueller_matrix

*function*

```python
def get_mueller_matrix(perpendicular: ArrayLike, parallel: ArrayLike) -> NDArray[np.float64]
```

Re-exported from: `particula.optics.polarization`

Construct the unnormalized sphere Mueller matrix from S1 and S2.

**Parameters**

- `perpendicular` (`ArrayLike`) — Complex perpendicular amplitude S1, dimensionless.
- `parallel` (`ArrayLike`) — Complex parallel amplitude S2, broadcastable with S1.

**Returns**

- (`NDArray[np.float64]`) — Real matrix with broadcast shape plus (4, 4), in the scattering
- (`NDArray[np.float64]`) — plane. Divide by medium wavenumber squared for m²/sr.

**Raises**

- `ValueError` — If either amplitude is nonfinite.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/polarization.py#L14-L45)

## particula.optics.get_phase_matrix

*function*

```python
def get_phase_matrix(differential_matrix: ArrayLike, scattering_cross_section: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]
```

Re-exported from: `particula.optics.cross_sections`

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

## particula.optics.get_population_mueller_matrix

*function*

```python
def get_population_mueller_matrix(differential_matrix: ArrayLike, number_density_weights: ArrayLike, *, particle_axis: int = -1) -> NDArray[np.float64]
```

Re-exported from: `particula.optics.population`

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

## particula.optics.get_population_optics

*function*

```python
def get_population_optics(cross_sections: OpticalCrossSections, asymmetry: ArrayLike, number_density_weights: ArrayLike, *, particle_axis: int = -1) -> PopulationOptics
```

Re-exported from: `particula.optics.population`

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

## particula.optics.get_population_weights

*function*

```python
def get_population_weights(concentration: ArrayLike, distribution_type: str, *, volume: ArrayLike | None = None, radius_grid: ArrayLike | None = None) -> NDArray[np.float64]
```

Re-exported from: `particula.optics.population`

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

## particula.optics.get_scattering_amplitudes

*function*

```python
def get_scattering_amplitudes(coefficients: MieCoefficients, angle: ArrayLike) -> ScatteringAmplitudes
```

Re-exported from: `particula.optics.angular`

Calculate S1 and S2 without repeating the coefficient calculation.

**Parameters**

- `coefficients` (`MieCoefficients`) — Zero-padded results from get_mie_coefficients.
- `angle` (`ArrayLike`) — Scalar or 1D scattering angles [rad] within [0, pi].

**Returns**

- (`ScatteringAmplitudes`) — Complex arrays shaped coefficient batch shape plus (n_angles,).
- (`ScatteringAmplitudes`) — Scalar angle retains a length-one axis. Inputs are unchanged.

**Raises**

- `ValueError` — If angles, coefficient shapes or values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/angular.py#L28-L80)

## particula.optics.rotate_stokes

*function*

```python
def rotate_stokes(stokes: ArrayLike, angle: ArrayLike) -> NDArray[np.float64]
```

Re-exported from: `particula.optics.polarization`

Express a Stokes vector in axes rotated toward the perpendicular axis.

**Parameters**

- `stokes` (`ArrayLike`) — Finite (..., 4) array ordered I,Q,U,V, in caller units.
- `angle` (`ArrayLike`) — Rotation [rad], broadcastable with the leading Stokes shape. Positive rotation uses Eparallel'=cos(angle)*Eparallel + sin(angle)*Eperpendicular.

**Returns**

- (`NDArray[np.float64]`) — New vectors with Q'=cos(2a)Q+sin(2a)U, U'=-sin(2a)Q+cos(2a)U.
- (`NDArray[np.float64]`) — I and V are unchanged. This operation does not normalize intensity.

**Raises**

- `ValueError` — If shapes or finite values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/polarization.py#L48-L75)

**Modules**

- `particula.optics.angular`
- `particula.optics.column`
- `particula.optics.cross_sections`
- `particula.optics.gpu`
- `particula.optics.mie`
- `particula.optics.polarization`
- `particula.optics.population`
