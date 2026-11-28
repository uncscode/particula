# particula.data.power

*module*

Read hourly NASA POWER surface shortwave irradiance from local JSON.

POWER's hourly point API labels the start of each hourly interval. The RE
community reports energy in Wh/m² for each hour. Dividing by its duration
in hours gives the hourly mean in W/m². The derived direct-normal, diffuse
horizontal and global horizontal fields retain their independent estimates.
See https://power.larc.nasa.gov/docs/services/api/temporal/hourly/.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L1-L1)

## particula.data.power.POWER_CHANNELS

*attribute* · *module attribute*

```python
POWER_CHANNELS = ('ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI', 'ALLSKY_SFC_SW_DIFF')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L20-L24)

## particula.data.power.PowerHour

*class* · *dataclass*

```python
class PowerHour
```

Independently reported mean surface irradiances in one NASA POWER hour.

**Attributes**

- `start_utc` (`datetime`) — UTC start of the one-hour averaging interval.
- `longitude_deg` (`float`) — Longitude of the response point [degrees east].
- `latitude_deg` (`float`) — Latitude of the response point [degrees north].
- `elevation_m` (`float`) — Provider-reported elevation [m].
- `global_horizontal` (`float`) — All-sky global horizontal irradiance [W m⁻²].
- `direct_normal` (`float`) — All-sky direct-normal irradiance [W m⁻²].
- `diffuse_horizontal` (`float`) — All-sky diffuse horizontal irradiance [W m⁻²].
- `api_version` (`str`) — POWER API version in the response header.
- `sha256` (`str`) — SHA-256 digest of the raw response file.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L27-L124)

### particula.data.power.PowerHour.api_version

*attribute* · *instance attribute*

```python
api_version: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L50-L50)

### particula.data.power.PowerHour.diffuse_horizontal

*attribute* · *instance attribute*

```python
diffuse_horizontal: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L49-L49)

### particula.data.power.PowerHour.direct_normal

*attribute* · *instance attribute*

```python
direct_normal: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L48-L48)

### particula.data.power.PowerHour.elevation_m

*attribute* · *instance attribute*

```python
elevation_m: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L46-L46)

### particula.data.power.PowerHour.global_horizontal

*attribute* · *instance attribute*

```python
global_horizontal: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L47-L47)

### particula.data.power.PowerHour.latitude_deg

*attribute* · *instance attribute*

```python
latitude_deg: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L45-L45)

### particula.data.power.PowerHour.longitude_deg

*attribute* · *instance attribute*

```python
longitude_deg: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L44-L44)

### particula.data.power.PowerHour.sha256

*attribute* · *instance attribute*

```python
sha256: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L51-L51)

### particula.data.power.PowerHour.start_utc

*attribute* · *instance attribute*

```python
start_utc: datetime
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L43-L43)

### particula.data.power.PowerHour.__init__

*method*

```python
def __init__(start_utc: datetime, longitude_deg: float, latitude_deg: float, elevation_m: float, global_horizontal: float, direct_normal: float, diffuse_horizontal: float, api_version: str, sha256: str) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L0-L0)

### particula.data.power.PowerHour.slab_boundary

*method*

```python
def slab_boundary(spectral_shape: SolarInput, *, direct_fraction: float, diffuse_fraction: float) -> SolarInput
```

Illuminate an additional slab with the POWER surface hourly mean.

Each component is distributed using a prescribed spectral shape. The
global horizontal field remains available for comparison. It does not
enter the normalization of the direct and diffuse components.

**Parameters**

- `spectral_shape` (`SolarInput`) — SolarInput supplying solar zenith cosine, wavelengths, integration rule and relative spectral shapes.
- `direct_fraction` (`float`) — Fraction in [0, 1] of broadband direct-normal irradiance assigned to the modeled wavelength range.
- `diffuse_fraction` (`float`) — Fraction in [0, 1] of broadband diffuse-horizontal irradiance assigned to the same wavelength range.

**Returns**

- (`SolarInput`) — Spectral densities [W m⁻² m⁻¹] on the supplied grid. Each component integrates to its POWER mean times its assigned fraction. Apply this boundary at the top of the added slab.

**Raises**

- `ValueError` — A fraction is invalid, a positive target has zero
spectral area, or a positive direct component has mu0 of zero.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L53-L124)

## particula.data.power._mean_irradiances

*function*

```python
def _mean_irradiances(data, key, fill_value)
```

Extract the three independently reported hourly means.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L192-L204)

## particula.data.power.read_power_hour

*function*

```python
def read_power_hour(path: str | Path, start: datetime, *, sha256: str | None = None) -> PowerHour
```

Read one hour of all-sky surface shortwave irradiance from local JSON.

**Parameters**

- `path` (`str | Path`) — Original NASA POWER hourly point API response in UTC with global horizontal, direct-normal and diffuse-horizontal fields.
- `start` (`datetime`) — Timezone-aware interval start aligned with a UTC hour.
- `sha256` (`str | None`) (default: `None`) — Optional expected SHA-256 hex digest of the raw bytes.

**Returns**

- (`PowerHour`) — Interval-mean irradiances [W m⁻²], location and provenance. Values in Wh m⁻² over one hour give the same numerical mean in W m⁻².

**Raises**

- `OSError` — The local file cannot be read.
- `ValueError` — The response, selected hour, time standard, field values,
units, coordinates or expected digest are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/power.py#L127-L189)
