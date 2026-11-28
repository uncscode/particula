# particula.optics.radiative_transfer

*module*

Plane-parallel shortwave transport by scalar discrete ordinates.

A matrix exponential solves a thin homogeneous slab. Matrix adding/doubling
avoids propagating growing upward solutions through optically thick layers.
See Hansen & Travis (1974), doi:10.1007/BF00168069, and Stamnes et al.
(1988), doi:10.1364/AO.27.002502. Azimuthal Fourier modes are independent
for the randomly oriented particles and isotropic surface textures used here.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L1-L1)

## particula.optics.radiative_transfer.ShortwaveFlux

*class* · *dataclass*

```python
class ShortwaveFlux
```

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

### particula.optics.radiative_transfer.ShortwaveFlux.band_bounds_m

*attribute* · *class attribute* · *instance attribute*

```python
band_bounds_m: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L53-L53)

### particula.optics.radiative_transfer.ShortwaveFlux.diffuse_down

*attribute* · *instance attribute*

```python
diffuse_down: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L47-L47)

### particula.optics.radiative_transfer.ShortwaveFlux.direct_down

*attribute* · *instance attribute*

```python
direct_down: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L46-L46)

### particula.optics.radiative_transfer.ShortwaveFlux.interface_height_m

*attribute* · *instance attribute*

```python
interface_height_m: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L45-L45)

### particula.optics.radiative_transfer.ShortwaveFlux.quadrature_weights_m

*attribute* · *instance attribute*

```python
quadrature_weights_m: np.ndarray | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L50-L50)

### particula.optics.radiative_transfer.ShortwaveFlux.radiance

*attribute* · *instance attribute*

```python
radiance: np.ndarray | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L49-L49)

### particula.optics.radiative_transfer.ShortwaveFlux.sources

*attribute* · *instance attribute*

```python
sources: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L51-L51)

### particula.optics.radiative_transfer.ShortwaveFlux.up

*attribute* · *instance attribute*

```python
up: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L48-L48)

### particula.optics.radiative_transfer.ShortwaveFlux.view

*attribute* · *instance attribute*

```python
view: tuple | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L52-L52)

### particula.optics.radiative_transfer.ShortwaveFlux.wavelength_m

*attribute* · *instance attribute*

```python
wavelength_m: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L44-L44)

### particula.optics.radiative_transfer.ShortwaveFlux.absorption

*property*

```python
absorption
```

Absorbed shortwave energy per unit area, time and wavelength.

**Returns**

- (`ndarray`) — Net downward flux at each layer top minus that at its bottom [W m⁻² m⁻¹], shape (layer, wavelength). Surface absorption is outside the atmospheric layers.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L67-L75)

### particula.optics.radiative_transfer.ShortwaveFlux.net_down

*property*

```python
net_down
```

Net downward irradiance density [W m⁻² m⁻¹] at each interface.

**Returns**

- (`ndarray`) — Direct plus diffuse downward flux minus upward flux, shape (interface, wavelength). Positive values carry energy downward.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L56-L64)

### particula.optics.radiative_transfer.ShortwaveFlux.__init__

*method*

```python
def __init__(wavelength_m: np.ndarray, interface_height_m: np.ndarray, direct_down: np.ndarray, diffuse_down: np.ndarray, up: np.ndarray, radiance: np.ndarray | None, quadrature_weights_m: np.ndarray | None, sources: tuple, view: tuple | None, band_bounds_m: np.ndarray | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L0-L0)

### particula.optics.radiative_transfer.ShortwaveFlux.difference

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

### particula.optics.radiative_transfer.ShortwaveFlux.integrate

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

## particula.optics.radiative_transfer._integrate_interval

*function*

```python
def _integrate_interval(wave, spectrum, low, high)
```

Integrate a sampled spectrum with linearly interpolated edges.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L176-L192)

## particula.optics.radiative_transfer._mode_interfaces

*function*

```python
def _mode_interfaces(slabs, surface_boundary, diffuse, direct, leaving, n)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L295-L322)

## particula.optics.radiative_transfer._phase_basis

*function*

```python
def _phase_basis(mu, order, count)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L195-L200)

## particula.optics.radiative_transfer._slab

*function*

```python
def _slab(depth, albedo, moments, basis, mu, weight, mu0)
```

Return reflection, transmission, beam sources and beam transmission.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L219-L255)

## particula.optics.radiative_transfer._surface_modes

*function*

```python
def _surface_modes(surface, mu, weight, mu0, modes, wave_slice)
```

Compute azimuth moments of the diffuse and direct surface response.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L258-L292)

## particula.optics.radiative_transfer._thin_exponential

*function*

```python
def _thin_exponential(generator, depth)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L203-L216)

## particula.optics.radiative_transfer._transport_optics

*function*

```python
def _transport_optics(column, index, streams, delta_m)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/radiative_transfer.py#L325-L343)

## particula.optics.radiative_transfer.sensor_band_radiance

*function*

```python
def sensor_band_radiance(result, response)
```

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

## particula.optics.radiative_transfer.solve_shortwave

*function*

```python
def solve_shortwave(column: SpectralColumn, sun: SolarInput, surface: Surface, *, streams=8, view_zenith_deg=None, relative_azimuth_deg=0.0, fourier_order=None, delta_m=True)
```

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
