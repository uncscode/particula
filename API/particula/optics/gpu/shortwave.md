# particula.optics.gpu.shortwave

*module*

Batched scalar shortwave transfer on a caller-selected Warp device.

The azimuthal quadrature and static BRDFs are prepared once. Device kernels
recompute layer scattering, adding/doubling, and interface fluxes as optical
properties change. See Hansen & Travis (1974), doi:10.1007/BF00168069.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L1-L1)

## particula.optics.gpu.shortwave.ShortwaveWorkspace

*class* · *dataclass*

```python
class ShortwaveWorkspace
```

Retained inputs, launch commands and latest results on one stream.

**Attributes**

- `layers` (`LayerData`) — Writable dimensionless depth, albedo (case, layer, wavelength) and normalized moments with an additional Legendre degree axis.
- `flux` (`wp.array`) — (case, interface, wavelength) vec3d of physical direct down, diffuse down and upwelling densities in W m⁻² m⁻¹.
- `broadband` (`wp.array | None`) — (case, interface) vec3d of those fluxes in W m⁻². None for a single wavelength without quadrature widths.
- `broadband_delta` (`wp.array | None`) — Optional (case, interface) vec3d of broadband case minus reference fluxes in W m⁻², in broadband field order.
- `radiance` (`wp.array | None`) — (case, wavelength) emergent W m⁻² sr⁻¹ m⁻¹ if a view was requested. Defined after enqueue and a successful check.
- `band_radiance` (`wp.array | None`) — Optional (case,) response mean in W m⁻² sr⁻¹ m⁻¹.
- `status` (`wp.array`) — (case, wavelength) integers. Zero means success, one invalid layer data, two a nonfinite result, negative an incomplete angular solve. Check before consuming device outputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L156-L245)

### particula.optics.gpu.shortwave.ShortwaveWorkspace._columns

*attribute* · *instance attribute*

```python
_columns: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L186-L186)

### particula.optics.gpu.shortwave.ShortwaveWorkspace._commands

*attribute* · *instance attribute*

```python
_commands: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L184-L184)

### particula.optics.gpu.shortwave.ShortwaveWorkspace._ready

*attribute* · *class attribute* · *instance attribute*

```python
_ready: bool = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L191-L191)

### particula.optics.gpu.shortwave.ShortwaveWorkspace._resources

*attribute* · *instance attribute*

```python
_resources: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L185-L185)

### particula.optics.gpu.shortwave.ShortwaveWorkspace._stream

*attribute* · *instance attribute*

```python
_stream: wp.Stream | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L190-L190)

### particula.optics.gpu.shortwave.ShortwaveWorkspace._suns

*attribute* · *instance attribute*

```python
_suns: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L187-L187)

### particula.optics.gpu.shortwave.ShortwaveWorkspace._surfaces

*attribute* · *instance attribute*

```python
_surfaces: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L188-L188)

### particula.optics.gpu.shortwave.ShortwaveWorkspace._view

*attribute* · *instance attribute*

```python
_view: tuple | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L189-L189)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.band_radiance

*attribute* · *instance attribute*

```python
band_radiance: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L182-L182)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.broadband

*attribute* · *instance attribute*

```python
broadband: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L179-L179)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.broadband_delta

*attribute* · *instance attribute*

```python
broadband_delta: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L180-L180)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.flux

*attribute* · *instance attribute*

```python
flux: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L178-L178)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.layers

*attribute* · *instance attribute*

```python
layers: LayerData
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L177-L177)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.radiance

*attribute* · *instance attribute*

```python
radiance: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L181-L181)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.status

*attribute* · *instance attribute*

```python
status: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L183-L183)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.__init__

*method*

```python
def __init__(layers: LayerData, flux: wp.array, broadband: wp.array | None, broadband_delta: wp.array | None, radiance: wp.array | None, band_radiance: wp.array | None, status: wp.array, _commands: tuple, _resources: tuple, _columns: tuple, _suns: tuple, _surfaces: tuple, _view: tuple | None, _stream: wp.Stream | None, _ready: bool = False) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L0-L0)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.check

*method*

```python
def check()
```

Synchronize and raise RuntimeError on missing or failed results.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L199-L210)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.download

*method*

```python
def download()
```

Download spectral results in case order.

**Returns**

- (`tuple[ShortwaveFlux, ...]`) — Detached CPU results for each case.

**Raises**

- `RuntimeError` — If no solve has run or a device result failed.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L212-L245)

### particula.optics.gpu.shortwave.ShortwaveWorkspace.enqueue

*method*

```python
def enqueue()
```

Queue transport on the prepared stream, overwriting prior results.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L193-L197)

## particula.optics.gpu.shortwave._angles

*function*

```python
def _angles(streams, fourier_order, relative_azimuth_deg, view_zenith_deg)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L264-L287)

## particula.optics.gpu.shortwave._column_layout

*function*

```python
def _column_layout(columns, sun, surface)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L248-L261)

## particula.optics.gpu.shortwave._device_stream

*function*

```python
def _device_stream(device, stream)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L310-L316)

## particula.optics.gpu.shortwave._integration

*function*

```python
def _integration(suns, sensor_response, has_view)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L49-L72)

## particula.optics.gpu.shortwave._moments

*function*

```python
def _moments(cols, nl, nw, streams, delta_m)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L299-L307)

## particula.optics.gpu.shortwave._quadrature

*function*

```python
def _quadrature(sun)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L40-L46)

## particula.optics.gpu.shortwave._reference_indices

*function*

```python
def _reference_indices(columns, suns, indices)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L75-L96)

## particula.optics.gpu.shortwave._sequence

*function*

```python
def _sequence(values, cls, count)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L33-L37)

## particula.optics.gpu.shortwave._tile_size

*function*

```python
def _tile_size(case_tile, lanes, n, nl)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L290-L296)

## particula.optics.gpu.shortwave._upload_boundary

*function*

```python
def _upload_boundary(suns, surfaces, mu, weight, modes, limit, device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L99-L153)

## particula.optics.gpu.shortwave.prepare_shortwave

*function*

```python
def prepare_shortwave(columns, sun, surface, *, device, stream=None, streams=8, view_zenith_deg=None, relative_azimuth_deg=0.0, fourier_order=None, delta_m=True, case_tile=None, reference_cases=None, sensor_response=None)
```

Prepare independent column/wavelength solves with reusable scratch.

**Parameters**

- `columns` (`SpectralColumn | Sequence[SpectralColumn]`) — One SpectralColumn or a sequence with equal wavelength grids and layer counts. Interface heights may vary between columns.
- `sun` (`SolarInput | Sequence[SolarInput]`) — SolarInput shared by all columns or one per column.
- `surface` (`Surface | Sequence[Surface]`) — Shared surface or a sequence of spectral surface boundaries.
- `device` (`str | warp.Device`) — Explicit Warp CPU or CUDA device. No automatic fallback.
- `stream` (`warp.Stream | None`) (default: `None`) — Optional CUDA stream on device. Input updates and consumers must use this stream or explicit inter-stream ordering.
- `streams` (`int`) (default: `8`) — Gauss–Legendre ordinates per hemisphere, 2 through 16.
- `view_zenith_deg` (`float | None`) (default: `None`) — Optional outgoing zenith angle [0, 90) in degrees.
- `relative_azimuth_deg` (`float`) (default: `0.0`) — Surface Sun–observer azimuth difference [deg].
- `fourier_order` (`int | None`) (default: `None`) — Maximum azimuthal harmonic, default 2 * streams - 1.
- `delta_m` (`bool`) (default: `True`) — Apply delta-M if a next phase moment is available.
- `case_tile` (`int | None`) (default: `None`) — Maximum concurrent column/wavelength lanes. The default targets 32 MiB of layer scratch and up to 4096 lanes.
- `reference_cases` (`ArrayLike | None`) (default: `None`) — Optional integer reference index for each case. Paired cases need matching heights and integration coordinates.
- `sensor_response` (`ArrayLike | None`) (default: `None`) — Optional nonnegative energy response on the grid. Requires a view and wavelength integration widths.

**Returns**

- (`ShortwaveWorkspace`) — Retained layer and boundary data. Use enqueue() after updating inputs. Use download() for detached CPU results.

**Raises**

- `ValueError` — If shapes, geometry, stream, or quadrature are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/shortwave.py#L319-L502)
