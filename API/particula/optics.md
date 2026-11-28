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

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L22-L33)

### particula.optics.ColumnOptics.layer_optical_depth

*attribute* · *instance attribute*

```python
layer_optical_depth: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L31-L31)

### particula.optics.ColumnOptics.optical_depth

*attribute* · *instance attribute*

```python
optical_depth: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L32-L32)

### particula.optics.ColumnOptics.transmission

*attribute* · *instance attribute*

```python
transmission: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L33-L33)

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

## particula.optics.OpticalConfig

*class* · *dataclass*

```python
class OpticalConfig
```

Re-exported from: `particula.optics.state`

Prescribed optical inputs for homogeneous spheres.

Wavelengths are ordered vacuum wavelengths [m]. Supply an absolute
particle index (scalar or per wavelength), or a species index table
(wavelength, material) in native particle-material order. Medium index
is real, positive and constant across the spectrum.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L50-L142)

### particula.optics.OpticalConfig.medium_index

*attribute* · *instance attribute*

```python
medium_index: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L61-L61)

### particula.optics.OpticalConfig.particle_index

*attribute* · *instance attribute*

```python
particle_index: tuple[complex, ...] | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L62-L62)

### particula.optics.OpticalConfig.species_index

*attribute* · *instance attribute*

```python
species_index: tuple[tuple[complex, ...], ...] | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L63-L63)

### particula.optics.OpticalConfig.wavelength

*attribute* · *instance attribute*

```python
wavelength: tuple[float, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L60-L60)

### particula.optics.OpticalConfig.__init__

*method*

```python
def __init__(wavelength, *, particle_index=None, species_index=None, medium_index=1.0)
```

Validate and freeze wavelengths and the chosen index model.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L65-L116)

### particula.optics.OpticalConfig.metadata

*method*

```python
def metadata()
```

Return JSON-compatible optical inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L126-L142)

### particula.optics.OpticalConfig.validate_materials

*method*

```python
def validate_materials(materials)
```

Check the species table against the native material axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L118-L124)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L17-L40)

### particula.optics.PopulationOptics.absorption

*attribute* · *instance attribute*

```python
absorption: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L34-L34)

### particula.optics.PopulationOptics.asymmetry

*attribute* · *instance attribute*

```python
asymmetry: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L38-L38)

### particula.optics.PopulationOptics.backscatter

*attribute* · *instance attribute*

```python
backscatter: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L35-L35)

### particula.optics.PopulationOptics.extinction

*attribute* · *instance attribute*

```python
extinction: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L32-L32)

### particula.optics.PopulationOptics.extinction_valid

*attribute* · *instance attribute*

```python
extinction_valid: NDArray[np.bool_]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L39-L39)

### particula.optics.PopulationOptics.radiation_pressure

*attribute* · *instance attribute*

```python
radiation_pressure: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L36-L36)

### particula.optics.PopulationOptics.scattering

*attribute* · *instance attribute*

```python
scattering: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L33-L33)

### particula.optics.PopulationOptics.scattering_valid

*attribute* · *instance attribute*

```python
scattering_valid: NDArray[np.bool_]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L40-L40)

### particula.optics.PopulationOptics.single_scattering_albedo

*attribute* · *instance attribute*

```python
single_scattering_albedo: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L37-L37)

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

## particula.optics.ShortwaveFlux

*class* · *dataclass*

```python
class ShortwaveFlux
```

Re-exported from: `particula.optics.radiative_transfer`

Spectral shortwave fluxes at every interface and optional top radiance.

**Attributes**

- `wavelength_m` (`np.ndarray`) — Increasing vacuum wavelengths [m].
- `interface_height_m` (`np.ndarray`) — Top-first interface heights [m].
- `direct_down` (`np.ndarray`) — Direct downward horizontal irradiance density [W m⁻² m⁻¹], shape (interface, wavelength).
- `diffuse_down` (`np.ndarray`) — Diffuse downward horizontal irradiance density on the same grid and in the same units as direct_down.
- `up` (`np.ndarray`) — Hemispheric upward irradiance density [W m⁻² m⁻¹] with axes (interface, wavelength).
- `radiance` (`np.ndarray | None`) — Optional spectral radiance [W m⁻² sr⁻¹ m⁻¹] leaving the top interface toward the requested observer, one per wavelength.
- `quadrature_weights_m` (`np.ndarray | None`) — Optional integration widths [m] per node.
- `sources` (`tuple`) — Provenance of solar input, surface and layer optics.
- `view` (`tuple | None`) — Optional (view zenith, solar-relative azimuth) in degrees.
- `band_bounds_m` (`np.ndarray | None`) — Optional quadrature band edges [m], shape (wavelength, 2).

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L22-L173)

### particula.optics.ShortwaveFlux.band_bounds_m

*attribute* · *class attribute* · *instance attribute*

```python
band_bounds_m: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L53-L53)

### particula.optics.ShortwaveFlux.diffuse_down

*attribute* · *instance attribute*

```python
diffuse_down: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L47-L47)

### particula.optics.ShortwaveFlux.direct_down

*attribute* · *instance attribute*

```python
direct_down: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L46-L46)

### particula.optics.ShortwaveFlux.interface_height_m

*attribute* · *instance attribute*

```python
interface_height_m: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L45-L45)

### particula.optics.ShortwaveFlux.quadrature_weights_m

*attribute* · *instance attribute*

```python
quadrature_weights_m: np.ndarray | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L50-L50)

### particula.optics.ShortwaveFlux.radiance

*attribute* · *instance attribute*

```python
radiance: np.ndarray | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L49-L49)

### particula.optics.ShortwaveFlux.sources

*attribute* · *instance attribute*

```python
sources: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L51-L51)

### particula.optics.ShortwaveFlux.up

*attribute* · *instance attribute*

```python
up: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L48-L48)

### particula.optics.ShortwaveFlux.view

*attribute* · *instance attribute*

```python
view: tuple | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L52-L52)

### particula.optics.ShortwaveFlux.wavelength_m

*attribute* · *instance attribute*

```python
wavelength_m: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L44-L44)

### particula.optics.ShortwaveFlux.absorption

*property*

```python
absorption
```

Absorbed shortwave energy per unit area, time and wavelength.

**Returns**

- (`ndarray`) — Net downward flux at each layer top minus that at its bottom [W m⁻² m⁻¹], shape (layer, wavelength). Surface absorption is outside the atmospheric layers.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L67-L75)

### particula.optics.ShortwaveFlux.net_down

*property*

```python
net_down
```

Net downward irradiance density [W m⁻² m⁻¹] at each interface.

**Returns**

- (`ndarray`) — Direct plus diffuse downward flux minus upward flux, shape (interface, wavelength). Positive values carry energy downward.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L56-L64)

### particula.optics.ShortwaveFlux.__init__

*method*

```python
def __init__(wavelength_m: np.ndarray, interface_height_m: np.ndarray, direct_down: np.ndarray, diffuse_down: np.ndarray, up: np.ndarray, radiance: np.ndarray | None, quadrature_weights_m: np.ndarray | None, sources: tuple, view: tuple | None, band_bounds_m: np.ndarray | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L0-L0)

### particula.optics.ShortwaveFlux.difference

*method*

```python
def difference(reference)
```

Calculate the signed effect of this case relative to a reference.

**Parameters**

- `reference` (`ShortwaveFlux`) — ShortwaveFlux with matching vertical and spectral coordinates, integration weights, band edges and view.

**Returns**

- (`ShortwaveFlux`) — Componentwise differences (this case minus reference), retaining spectral densities and both source records.

**Raises**

- `ValueError` — Coordinates, integration metadata or views differ.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L134-L173)

### particula.optics.ShortwaveFlux.integrate

*method*

```python
def integrate(spectrum, *, band_m=None)
```

Integrate a spectral density over the grid or a selected interval.

**Parameters**

- `spectrum` (`ArrayLike`) — Spectral densities with wavelength on the final axis. An irradiance input in W m⁻² m⁻¹ integrates to W m⁻².
- `band_m` (`tuple[float, float] | None`) (default: `None`) — Optional (lower, upper) wavelength limits [m]. Sampled spectra use trapezoids with linearly interpolated endpoints. For representative nodes the limits must match native band edges, and the integration uses the supplied quadrature widths.

**Returns**

- (`numpy.floating | ndarray`) — Integral over wavelength. All leading input axes are preserved.

**Raises**

- `ValueError` — The spectral axis or interval is invalid, a sampled
grid lacks spectral width, or quadrature band edges are absent
or incompatible with the requested interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L77-L132)

## particula.optics.SolarInput

*class* · *dataclass*

```python
class SolarInput
```

Re-exported from: `particula.optics.solar`

Collimated and isotropic diffuse illumination at the top of a column.

**Attributes**

- `wavelength_m` (`np.ndarray`) — Increasing positive vacuum wavelengths [m].
- `mu0` (`float`) — Solar zenith cosine in [0, 1]. The solver uses zero for a Sun at or below the horizon and suppresses the collimated beam.
- `direct_normal` (`np.ndarray`) — Incident beam irradiance density [W m⁻² m⁻¹], measured normal to the ray. Its horizontal component is mu0 times direct_normal.
- `diffuse_horizontal` (`np.ndarray`) — Downward diffuse irradiance density [W m⁻² m⁻¹] on a horizontal plane. Incident diffuse radiance is diffuse_horizontal divided by pi, uniform over the hemisphere.
- `quadrature_weights_m` (`np.ndarray | None`) — Optional effective integration widths [m] at the representative wavelengths. Without weights the grid is sampled.
- `source` (`str`) — Provenance of the illumination and spectral distribution.
- `band_bounds_m` (`np.ndarray | None`) — Optional lower and upper wavelengths [m] for each quadrature node, shape (wavelength, 2).

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L30-L118)

### particula.optics.SolarInput.band_bounds_m

*attribute* · *class attribute* · *instance attribute*

```python
band_bounds_m: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L57-L57)

### particula.optics.SolarInput.diffuse_horizontal

*attribute* · *instance attribute*

```python
diffuse_horizontal: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L54-L54)

### particula.optics.SolarInput.direct_normal

*attribute* · *instance attribute*

```python
direct_normal: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L53-L53)

### particula.optics.SolarInput.mu0

*attribute* · *instance attribute*

```python
mu0: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L52-L52)

### particula.optics.SolarInput.quadrature_weights_m

*attribute* · *class attribute* · *instance attribute*

```python
quadrature_weights_m: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L55-L55)

### particula.optics.SolarInput.source

*attribute* · *class attribute* · *instance attribute*

```python
source: str = 'user upper boundary'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L56-L56)

### particula.optics.SolarInput.wavelength_m

*attribute* · *instance attribute*

```python
wavelength_m: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L51-L51)

### particula.optics.SolarInput.__init__

*method*

```python
def __init__(wavelength_m: ArrayLike, mu0: float, direct_normal: ArrayLike, diffuse_horizontal: ArrayLike, quadrature_weights_m: ArrayLike | None = None, source: str = 'user upper boundary', band_bounds_m: ArrayLike | None = None)
```

Copy spectral boundary data onto a common read-only wavelength grid.

**Parameters**

- `wavelength_m` (`ArrayLike`) — One-dimensional increasing positive wavelengths [m].
- `mu0` (`float`) — Scalar cosine of the solar zenith angle in [0, 1].
- `direct_normal` (`ArrayLike`) — Nonnegative beam density [W m⁻² m⁻¹], a scalar or one value per wavelength.
- `diffuse_horizontal` (`ArrayLike`) — Nonnegative diffuse density [W m⁻² m⁻¹], a scalar or one value per wavelength.
- `quadrature_weights_m` (`ArrayLike | None`) (default: `None`) — Optional positive integration widths [m], scalar or one per wavelength.
- `source` (`str`) (default: `'user upper boundary'`) — Description of the source of the boundary data.
- `band_bounds_m` (`ArrayLike | None`) (default: `None`) — Optional pairs of ordered band edges [m], one pair per wavelength. Each pair must contain its quadrature node.

**Raises**

- `ValueError` — Coordinates, cosine or spectral fields are invalid,
or band bounds lack quadrature weights.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L59-L118)

## particula.optics.SpectralColumn

*class* · *dataclass*

```python
class SpectralColumn
```

Re-exported from: `particula.optics.column`

Vertical extinction, scattering and phase function of homogeneous layers.

**Attributes**

- `wavelength_m` (`NDArray`) — Strictly increasing positive vacuum wavelengths [m].
- `interface_height_m` (`NDArray`) — Interface heights [m] in strictly descending order, including both the top and bottom of the column.
- `optical_depth` (`NDArray`) — Nonnegative vertical extinction optical depth [dimensionless], shape (layer, wavelength).
- `single_scattering_albedo` (`NDArray`) — Scattering fraction of extinction in [0, 1] with the same axes as optical_depth.
- `phase_moments` (`NDArray`) — Normalized Legendre coefficients of the phase function, shape (layer, wavelength, order). Each lies in [-1, 1]. Order zero equals one wherever scattering depth is positive. Order one is the mean cosine of the scattering angle.
- `sources` (`tuple[str, ...]`) — Provenance of the optical properties and vertical grid.

**Raises**

- `ValueError` — Input arrays are nonfinite, the axes disagree, or a
coordinate or optical property violates its physical domain.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L114-L323)

### particula.optics.SpectralColumn.interface_height_m

*attribute* · *instance attribute*

```python
interface_height_m: NDArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L138-L138)

### particula.optics.SpectralColumn.optical_depth

*attribute* · *instance attribute*

```python
optical_depth: NDArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L139-L139)

### particula.optics.SpectralColumn.phase_moments

*attribute* · *instance attribute*

```python
phase_moments: NDArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L141-L141)

### particula.optics.SpectralColumn.single_scattering_albedo

*attribute* · *instance attribute*

```python
single_scattering_albedo: NDArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L140-L140)

### particula.optics.SpectralColumn.sources

*attribute* · *class attribute* · *instance attribute*

```python
sources: tuple[str, ...] = ('user layer optics and interface heights')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L142-L142)

### particula.optics.SpectralColumn.wavelength_m

*attribute* · *instance attribute*

```python
wavelength_m: NDArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L137-L137)

### particula.optics.SpectralColumn.__init__

*method*

```python
def __init__(wavelength_m: NDArray, interface_height_m: NDArray, optical_depth: NDArray, single_scattering_albedo: NDArray, phase_moments: NDArray, sources: tuple[str, ...] = ('user layer optics and interface heights')) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L0-L0)

### particula.optics.SpectralColumn.__post_init__

*method*

```python
def __post_init__()
```

Freeze and validate the layer and wavelength axes.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L144-L174)

### particula.optics.SpectralColumn.from_population

*method* · *classmethod*

```python
def from_population(wavelength_m, interface_height_m, population, box_indices=None, *, hg_order=32)
```

Convert homogeneous population optics to vertical layer optics.

Extinction times geometrical thickness gives vertical optical depth.
The Henyey–Greenstein closure sets the moment of order l to g**l, where
g is the population asymmetry parameter.

**Parameters**

- `wavelength_m` (`ArrayLike`) — Increasing vacuum wavelengths [m] of the population optics, shape (wavelength,).
- `interface_height_m` (`ArrayLike`) — Strictly descending interface heights [m].
- `population` (`PopulationOptics`) — Bulk optics with extinction [m⁻¹], single scattering albedo and asymmetry (both dimensionless), each with axes (box, wavelength).
- `box_indices` (`ArrayLike | None`) (default: `None`) — Integer box index for each layer, in top-first order. If omitted, each row of population optics supplies one layer.
- `hg_order` (`int`) (default: `32`) — Highest phase moment to retain, an integer from 1 to 255.

**Returns**

- (`SpectralColumn`) — One layer per adjacent pair of interfaces and hg_order + 1 phase moments per layer and wavelength.

**Raises**

- `ValueError` — The grids, extinction, optical properties, box mapping
or requested phase order are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L176-L251)

### particula.optics.SpectralColumn.plus

*method*

```python
def plus(other)
```

Combine independent optical constituents in the same layers.

**Parameters**

- `other` (`SpectralColumn`) — Another SpectralColumn on exactly the same wavelength and interface-height grids.

**Returns**

- (`SpectralColumn`) — New column with summed extinction and scattering depths. The phase moments are weighted by scattering optical depth. Missing higher orders contribute zero. Transparent layers receive albedo zero.

**Raises**

- `ValueError` — The two columns have different coordinates.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L276-L323)

### particula.optics.SpectralColumn.scaled

*method*

```python
def scaled(factor)
```

Multiply all layer optical depths by a common factor.

**Parameters**

- `factor` (`float`) — Finite nonnegative dimensionless extinction multiplier.

**Returns**

- (`SpectralColumn`) — New column with the same albedos and phase functions. Multiplying by zero gives a transparent column on the original grid.

**Raises**

- `ValueError` — The multiplier is nonscalar, nonfinite or negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L253-L274)

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

## particula.optics.dusty_ground

*function*

```python
def dusty_ground(wavelength_m)
```

Re-exported from: `particula.optics.surface`

Construct an illustrative dry mineral-soil Lambertian surface.

**Parameters**

- `wavelength_m` (`ArrayLike`) — Increasing vacuum wavelengths [m] within 280–4000 nm.

**Returns**

- (`Surface`) — Hemispheric albedo interpolated linearly between eight prescribed spectral knots. The curve represents a reference scenario.

**Raises**

- `ValueError` — The wavelength grid is invalid or exceeds the reference.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L241-L260)

## particula.optics.earth_reference

*function*

```python
def earth_reference(latitude_deg, longitude_deg, time, interface_height_m, wavelength_m=None, *, grid='reduced', pressure_pa=None, gas_optical_depth=None, profile_name='user profile')
```

Re-exported from: `particula.optics.solar`

Build an offline Earth reference atmosphere and its solar illumination.

The highest interface receives extraterrestrial TSIS-1 sunlight corrected
for Earth–Sun distance. Include the overlying atmosphere when interpreting
surface or top-of-atmosphere flux. Geographic location sets solar geometry.
The default molecular profile is the AFGL US standard atmosphere.

**Parameters**

- `latitude_deg` (`float`) — Geodetic latitude [degrees north], in [-90, 90].
- `longitude_deg` (`float`) — Longitude [degrees east], in [-180, 180].
- `time` (`datetime`) — Timezone-aware instant of illumination.
- `interface_height_m` (`ArrayLike`) — Descending interface altitudes above sea level [m], from −500 to 120000 m. The lowest interface sets surface elevation.
- `wavelength_m` (`ArrayLike | None`) (default: `None`) — Optional increasing wavelengths [m] within the gas table. Its first band starts at approximately 280.033604 nm and the last ends at 4000 nm. A custom grid uses the containing band's effective gas depth, matched to vertical direct transmission. Native nodes use quadrature that conserves the TSIS-1 energy in each band.
- `grid` (`str`) (default: `'reduced'`) — Native REPTRAN gas quadrature, "reduced" or "full". Custom wavelengths use the full table to derive their band depths.
- `pressure_pa` (`ArrayLike | None`) (default: `None`) — Optional replacement pressures [Pa], one per interface, increasing downward. They determine Rayleigh optical depth.
- `gas_optical_depth` (`ArrayLike | None`) (default: `None`) — Optional replacement vertical absorption depths [dimensionless], shape (layer, wavelength).
- `profile_name` (`str`) (default: `'user profile'`) — Source label for supplied pressures or gas depths.

**Returns**

- (`EarthReference`) — Molecular optics, Sun position, incident spectrum and field provenance. Incident diffuse irradiance is zero.

**Raises**

- `ValueError` — Location, time, physical fields or grids are invalid or
exceed the coverage of the reference data.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L477-L604)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L36-L76)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/column.py#L79-L111)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L187-L229)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L115-L184)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/population.py#L53-L112)

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

## particula.optics.lambertian_surface

*function*

```python
def lambertian_surface(wavelength_m, albedo, *, source='prescribed Lambertian surface')
```

Re-exported from: `particula.optics.surface`

Construct a diffusely reflecting surface with a prescribed albedo.

**Parameters**

- `wavelength_m` (`ArrayLike`) — Increasing positive vacuum wavelengths [m].
- `albedo` (`ArrayLike`) — Hemispheric albedo in [0, 1], scalar or one per wavelength.
- `source` (`str`) (default: `'prescribed Lambertian surface'`) — Provenance of the prescribed albedo.

**Returns**

- (`Surface`) — Angle-independent BRDF equal to albedo / pi [sr⁻¹].

**Raises**

- `ValueError` — The coordinates, albedo range or spectral shape is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L149-L170)

## particula.optics.mix_refractive_index

*function*

```python
def mix_refractive_index(index, volume_fraction)
```

Re-exported from: `particula.optics.state`

Mix absolute indices along the last axis on a volume basis.

The fractions are dimensionless, nonnegative and sum to one.
Both inputs broadcast with their common final material axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L33-L47)

## particula.optics.molecular_column

*function*

```python
def molecular_column(wavelength_m, interface_height_m, pressure_pa, gas_optical_depth, *, source='user profile')
```

Re-exported from: `particula.optics.solar`

Combine Rayleigh scattering and prescribed gas absorption in each layer.

The Hansen–Travis sea-level Rayleigh depth is scaled by the layer
pressure difference divided by 101325 Pa. The scalar Rayleigh phase
function uses normalized Legendre moments (1, 0, 0.1).

**Parameters**

- `wavelength_m` (`ArrayLike`) — Increasing positive vacuum wavelengths [m].
- `interface_height_m` (`ArrayLike`) — Strictly descending interface altitudes [m].
- `pressure_pa` (`ArrayLike`) — Nonnegative interface pressures [Pa], increasing downward.
- `gas_optical_depth` (`ArrayLike`) — Nonnegative vertical absorption optical depth [dimensionless], shape (layer, wavelength).
- `source` (`str`) (default: `'user profile'`) — Provenance of the pressure profile and gas absorption.

**Returns**

- (`SpectralColumn`) — Total extinction, scattering albedo and moments.

**Raises**

- `ValueError` — Coordinates or optical depths are invalid or incompatible.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L411-L474)

## particula.optics.ocean_surface

*function*

```python
def ocean_surface(wavelength_m, *, wind_speed=None, refractive_index=1.34, leaving_radiance=None)
```

Re-exported from: `particula.optics.surface`

Construct a dark-water reference or a wind-roughened glint surface.

The Cox–Munk option uses an isotropic Gaussian slope distribution, Smith
masking and unpolarized Fresnel reflection at the air–water interface.

**Parameters**

- `wavelength_m` (`ArrayLike`) — Increasing positive vacuum wavelengths [m]. The Lambertian reference curve spans 280–4000 nm.
- `wind_speed` (`float | None`) (default: `None`) — Surface wind speed [m s⁻¹] in [1, 14] for Cox–Munk. When omitted, use the prescribed dark-water Lambertian spectrum.
- `refractive_index` (`ArrayLike`) (default: `1.34`) — Real water refractive index above one, scalar or one per wavelength. Used only with the wind-roughened model.
- `leaving_radiance` (`ArrayLike | None`) (default: `None`) — Optional isotropic outgoing water-leaving radiance [W m⁻² sr⁻¹ m⁻¹], nonnegative, scalar or one per wavelength.

**Returns**

- (`Surface`) — The selected reflection law and optional upward source.

**Raises**

- `ValueError` — Wavelengths or parameters for the chosen model are
invalid, outside its domain or incompatible in shape.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L181-L238)

## particula.optics.particle_optics

*function*

```python
def particle_optics(particles, config)
```

Re-exported from: `particula.optics.state`

Return per-particle Mie efficiencies and cross sections.

Arrays have axes (box, wavelength, slot). Empty slots have zero radius
for the optical calculation. Discrete and resolved sizes come from
material mass and density. Continuous PDFs use their explicit nodes.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L145-L185)

## particula.optics.population_optics

*function*

```python
def population_optics(particles, config) -> PopulationOptics
```

Re-exported from: `particula.optics.state`

Evaluate live CPU particle state into (box, wavelength) results.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/state.py#L188-L212)

## particula.optics.ross_li_surface

*function*

```python
def ross_li_surface(wavelength_m, isotropic, volumetric, geometric, *, source='user Ross–Li')
```

Re-exported from: `particula.optics.surface`

Construct a land BRDF from MODIS RossThick/LiSparseReciprocal kernels.

**Parameters**

- `wavelength_m` (`ArrayLike`) — Increasing positive vacuum wavelengths [m].
- `isotropic` (`ArrayLike`) — Isotropic reflectance coefficient, scalar or one value per wavelength. All three coefficients are finite and dimensionless.
- `volumetric` (`ArrayLike`) — RossThick volumetric coefficient on the same grid.
- `geometric` (`ArrayLike`) — LiSparseReciprocal geometric coefficient on the same grid.
- `source` (`str`) (default: `'user Ross–Li'`) — Provenance of the coefficients, such as an MCD43 retrieval.

**Returns**

- (`Surface`) — Land boundary with BRDF [sr⁻¹] equal to the kernel-weighted sum divided by pi. The caller supplies coefficients applicable to the requested geometry.

**Raises**

- `ValueError` — The coordinate or spectral coefficients are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L263-L293)

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

## particula.optics.sensor_band_radiance

*function*

```python
def sensor_band_radiance(result, response)
```

Re-exported from: `particula.optics.radiative_transfer`

Calculate the energy-response-weighted mean top-of-column radiance.

**Parameters**

- `result` (`ShortwaveFlux`) — ShortwaveFlux with a directional radiance spectrum.
- `response` (`ArrayLike`) — Nonnegative energy response, one value per solved wavelength. Its normalization cancels in the mean. For a photon-counting detector supply the calibrated energy response or include the instrument's required photon weighting.

**Returns**

- (`numpy.floating`) — Band-mean radiance density [W m⁻² sr⁻¹ m⁻¹], the integral of radiance times response divided by the integrated response.

**Raises**

- `ValueError` — Radiance is absent, the response has invalid shape,
sign or values, or the integrated response lacks positive width.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L505-L537)

## particula.optics.solar_position

*function*

```python
def solar_position(latitude_deg, longitude_deg, time: datetime)
```

Re-exported from: `particula.optics.solar`

Calculate geometric solar position and Earth–Sun distance at an instant.

**Parameters**

- `latitude_deg` (`float`) — Geodetic latitude [degrees north], in [-90, 90].
- `longitude_deg` (`float`) — Longitude [degrees east], in [-180, 180].
- `time` (`datetime`) — Timezone-aware datetime, converted to UTC for the ephemeris.

**Returns**

- (`SolarPosition`) — Zenith and azimuth angles [degrees] and distance [au]. The horizon is horizontal and atmospheric refraction is omitted.

**Raises**

- `ValueError` — Coordinates lie outside their domains or time is naive.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L137-L207)

## particula.optics.solve_shortwave

*function*

```python
def solve_shortwave(column: SpectralColumn, sun: SolarInput, surface: Surface, *, streams=8, view_zenith_deg=None, relative_azimuth_deg=0.0, fourier_order=None, delta_m=True)
```

Re-exported from: `particula.optics.radiative_transfer`

Solve scalar plane-parallel shortwave transfer through a layered column.

The discrete-ordinate solver includes multiple scattering and surface
reflection. Returned direct irradiance uses the unscaled extinction depth.
With delta-M scaling, unresolved forward-scattered light contributes to
diffuse downward irradiance.

**Parameters**

- `column` (`SpectralColumn`) — Top-first layer optical properties on a common wavelength grid.
- `sun` (`SolarInput`) — Spectral direct-normal and diffuse-horizontal irradiance densities [W m⁻² m⁻¹] at the upper interface, with the solar zenith cosine.
- `surface` (`Surface`) — Reflecting lower boundary on the column wavelength grid.
- `streams` (`int`) (default: `8`) — Number of Gauss–Legendre ordinates per hemisphere, 2–32.
- `view_zenith_deg` (`float | None`) (default: `None`) — Optional angle [degrees] from the upward normal to the observer in [0, 90). A zero-weight ordinate resolves this view.
- `relative_azimuth_deg` (`float`) (default: `0.0`) — Azimuth [degrees] between surface-to-Sun and surface-to-observer directions. At 180 the observer lies across the surface normal from the Sun, in the specular plane.
- `fourier_order` (`int | None`) (default: `None`) — Highest azimuthal harmonic for a directional solve, 0–64. The default is 2 * streams - 1. Flux alone uses order zero.
- `delta_m` (`bool`) (default: `True`) — Apply delta-M scaling to represent a forward lobe if a phase moment beyond the retained orders is available.

**Returns**

- (`ShortwaveFlux`) — Spectral irradiance densities [W m⁻² m⁻¹] at each interface and optionally the spectral radiance leaving the column top [W m⁻² sr⁻¹ m⁻¹] in the requested viewing direction.

**Raises**

- `ValueError` — Spectral grids differ or the stream count, viewing
geometry or Fourier order is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L346-L502)

**Modules**

- `particula.optics._spectral`
- `particula.optics.angular`
- `particula.optics.column`
- `particula.optics.cross_sections`
- `particula.optics.gpu`
- `particula.optics.humidity`
- `particula.optics.mie`
- `particula.optics.polarization`
- `particula.optics.population`
- `particula.optics.radiative_transfer`
- `particula.optics.retrieval`
- `particula.optics.solar`
- `particula.optics.state`
- `particula.optics.surface`
