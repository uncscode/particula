# particula.optics.surface

*module*

Scalar shortwave reflection over horizontally uniform land and ocean.

The RossThick/LiSparseReciprocal kernels follow the MODIS MCD43 algorithm.
The idealized wind-roughened ocean uses the isotropic Cox–Munk mean square
slope (1954, doi:10.1364/JOSA.44.000838) and unpolarized Fresnel reflectance.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L1-L1)

## particula.optics.surface.Surface

*class* · *dataclass*

```python
class Surface
```

Spectral reflection model and optional emission at the lower interface.

Construct instances with the surface factories to validate their inputs.

**Attributes**

- `wavelength_m` (`np.ndarray`) — Increasing positive vacuum wavelengths [m].
- `model` (`str`) — Angular law, "lambertian", "ross_li" or "cox_munk".
- `coefficients` (`tuple`) — Model-specific parameters. Lambertian uses one albedo spectrum. Ross–Li uses isotropic, volumetric and geometric spectral coefficients. Cox–Munk uses a real water refractive-index spectrum and the scalar mean square slope of the facets.
- `source` (`str`) — Provenance of the surface model or retrieval.
- `leaving_radiance` (`np.ndarray | None`) — Optional wavelength-dependent isotropic upward source radiance [W m⁻² sr⁻¹ m⁻¹], added to reflected light.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L15-L110)

### particula.optics.surface.Surface.coefficients

*attribute* · *instance attribute*

```python
coefficients: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L35-L35)

### particula.optics.surface.Surface.leaving_radiance

*attribute* · *class attribute* · *instance attribute*

```python
leaving_radiance: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L37-L37)

### particula.optics.surface.Surface.model

*attribute* · *instance attribute*

```python
model: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L34-L34)

### particula.optics.surface.Surface.source

*attribute* · *instance attribute*

```python
source: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L36-L36)

### particula.optics.surface.Surface.wavelength_m

*attribute* · *instance attribute*

```python
wavelength_m: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L33-L33)

### particula.optics.surface.Surface.__init__

*method*

```python
def __init__(wavelength_m: np.ndarray, model: str, coefficients: tuple, source: str, leaving_radiance: np.ndarray | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L0-L0)

### particula.optics.surface.Surface.brdf

*method*

```python
def brdf(mu_in, mu_out, relative_azimuth_rad)
```

Evaluate the spectral BRDF for a set of directions.

**Parameters**

- `mu_in` (`ArrayLike`) — Cosine of the incidence zenith angle in (0, 1], measured from the surface normal toward the light source.
- `mu_out` (`ArrayLike`) — Cosine of the emergence zenith angle in (0, 1], measured from the surface normal toward the observer.
- `relative_azimuth_rad` (`ArrayLike`) — Azimuth difference [rad] of those directions. Zero places the observer on the source side of the surface. Specular reflection on a horizontal facet occurs at pi. All three angular inputs broadcast to a common shape.

**Returns**

- (`ndarray`) — BRDF [sr⁻¹] with wavelength first and the broadcast geometry axes following. The independent water-leaving radiance is excluded.

**Raises**

- `ValueError` — The directions are nonfinite, cosines lie outside
(0, 1], or the angular inputs cannot broadcast together.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L39-L110)

## particula.optics.surface._dot

*function*

```python
def _dot(mu_a, mu_b, phi)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L120-L125)

## particula.optics.surface._land_kernels

*function*

```python
def _land_kernels(mu_a, mu_b, phi)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L128-L146)

## particula.optics.surface._reference_albedo

*function*

```python
def _reference_albedo(wave, values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L173-L178)

## particula.optics.surface._shadow

*function*

```python
def _shadow(mu, variance)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L113-L117)

## particula.optics.surface.dusty_ground

*function*

```python
def dusty_ground(wavelength_m)
```

Construct an illustrative dry mineral-soil Lambertian surface.

**Parameters**

- `wavelength_m` (`ArrayLike`) — Increasing vacuum wavelengths [m] within 280–4000 nm.

**Returns**

- (`Surface`) — Hemispheric albedo interpolated linearly between eight prescribed spectral knots. The curve represents a reference scenario.

**Raises**

- `ValueError` — The wavelength grid is invalid or exceeds the reference.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/surface.py#L241-L260)

## particula.optics.surface.lambertian_surface

*function*

```python
def lambertian_surface(wavelength_m, albedo, *, source='prescribed Lambertian surface')
```

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

## particula.optics.surface.ocean_surface

*function*

```python
def ocean_surface(wavelength_m, *, wind_speed=None, refractive_index=1.34, leaving_radiance=None)
```

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

## particula.optics.surface.ross_li_surface

*function*

```python
def ross_li_surface(wavelength_m, isotropic, volumetric, geometric, *, source='user Ross–Li')
```

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
