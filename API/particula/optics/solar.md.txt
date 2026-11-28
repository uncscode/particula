# particula.optics.solar

*module*

Offline shortwave illumination and an idealized Earth reference atmosphere.

Solar position uses the apparent solar longitude and equation of time
(Meeus, Astronomical Algorithms, ch. 25). Solar zenith is geometrical.
The reference combines the TSIS-1 spectrum (Coddington et al.,
doi:10.1029/2022EA002637) with REPTRAN gas absorption (Gasteiger et al.,
doi:10.1016/j.jqsrt.2014.06.024) in the AFGL US standard atmosphere.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L1-L1)

## particula.optics.solar._DATA

*attribute* · *module attribute*

```python
_DATA = Path(__file__).with_name('reference_data')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L27-L27)

## particula.optics.solar.EarthReference

*class* · *dataclass*

```python
class EarthReference
```

Clear reference column and extraterrestrial illumination at a location.

**Attributes**

- `column` (`SpectralColumn`) — Molecular optical properties on the requested height grid.
- `sun` (`SolarInput`) — Spectral incident sunlight at the uppermost interface.
- `position` (`SolarPosition`) — Geometric solar zenith, azimuth and Earth–Sun distance.
- `band_bounds_m` (`np.ndarray | None`) — Containing REPTRAN band edges [m], shape (wavelength, 2). On a custom grid these identify the gas approximation used for each sampled wavelength.
- `sources` (`tuple`) — Provenance of the grid, atmosphere and illumination.
- `latitude_deg` (`float`) — Geodetic latitude [degrees north].
- `longitude_deg` (`float`) — Longitude [degrees east].
- `time` (`datetime`) — Timezone-aware instant specified by the caller.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L237-L261)

### particula.optics.solar.EarthReference.band_bounds_m

*attribute* · *instance attribute*

```python
band_bounds_m: np.ndarray | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L257-L257)

### particula.optics.solar.EarthReference.column

*attribute* · *instance attribute*

```python
column: SpectralColumn
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L254-L254)

### particula.optics.solar.EarthReference.latitude_deg

*attribute* · *instance attribute*

```python
latitude_deg: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L259-L259)

### particula.optics.solar.EarthReference.longitude_deg

*attribute* · *instance attribute*

```python
longitude_deg: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L260-L260)

### particula.optics.solar.EarthReference.position

*attribute* · *instance attribute*

```python
position: SolarPosition
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L256-L256)

### particula.optics.solar.EarthReference.sources

*attribute* · *instance attribute*

```python
sources: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L258-L258)

### particula.optics.solar.EarthReference.sun

*attribute* · *instance attribute*

```python
sun: SolarInput
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L255-L255)

### particula.optics.solar.EarthReference.time

*attribute* · *instance attribute*

```python
time: datetime
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L261-L261)

### particula.optics.solar.EarthReference.__init__

*method*

```python
def __init__(column: SpectralColumn, sun: SolarInput, position: SolarPosition, band_bounds_m: np.ndarray | None, sources: tuple, latitude_deg: float, longitude_deg: float, time: datetime) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L0-L0)

## particula.optics.solar.SolarInput

*class* · *dataclass*

```python
class SolarInput
```

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

### particula.optics.solar.SolarInput.band_bounds_m

*attribute* · *class attribute* · *instance attribute*

```python
band_bounds_m: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L57-L57)

### particula.optics.solar.SolarInput.diffuse_horizontal

*attribute* · *instance attribute*

```python
diffuse_horizontal: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L54-L54)

### particula.optics.solar.SolarInput.direct_normal

*attribute* · *instance attribute*

```python
direct_normal: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L53-L53)

### particula.optics.solar.SolarInput.mu0

*attribute* · *instance attribute*

```python
mu0: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L52-L52)

### particula.optics.solar.SolarInput.quadrature_weights_m

*attribute* · *class attribute* · *instance attribute*

```python
quadrature_weights_m: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L55-L55)

### particula.optics.solar.SolarInput.source

*attribute* · *class attribute* · *instance attribute*

```python
source: str = 'user upper boundary'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L56-L56)

### particula.optics.solar.SolarInput.wavelength_m

*attribute* · *instance attribute*

```python
wavelength_m: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L51-L51)

### particula.optics.solar.SolarInput.__init__

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

## particula.optics.solar.SolarPosition

*class* · *dataclass*

```python
class SolarPosition
```

Solar direction at a location and Earth–Sun distance at that instant.

**Attributes**

- `zenith_deg` (`float`) — Geometric zenith angle [degrees], zero directly overhead and greater than 90 below the horizon.
- `azimuth_deg` (`float`) — Bearing [degrees] clockwise from north on [0, 360).
- `distance_au` (`float`) — Earth–Sun distance [astronomical units].

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L121-L134)

### particula.optics.solar.SolarPosition.azimuth_deg

*attribute* · *instance attribute*

```python
azimuth_deg: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L133-L133)

### particula.optics.solar.SolarPosition.distance_au

*attribute* · *instance attribute*

```python
distance_au: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L134-L134)

### particula.optics.solar.SolarPosition.zenith_deg

*attribute* · *instance attribute*

```python
zenith_deg: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L132-L132)

### particula.optics.solar.SolarPosition.__init__

*method*

```python
def __init__(zenith_deg: float, azimuth_deg: float, distance_au: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L0-L0)

## particula.optics.solar._extraterrestrial

*function* · *cached*

```python
def _extraterrestrial()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L210-L212)

## particula.optics.solar._gas_band_equivalent

*function*

```python
def _gas_band_equivalent(data, depth, wave)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L313-L336)

## particula.optics.solar._gas_reference

*function* · *cached*

```python
def _gas_reference(grid)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L264-L270)

## particula.optics.solar._profile_grid

*function*

```python
def _profile_grid(data, height)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L282-L310)

## particula.optics.solar._solar_band_energy

*function*

```python
def _solar_band_energy(bounds)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L273-L279)

## particula.optics.solar.earth_reference

*function*

```python
def earth_reference(latitude_deg, longitude_deg, time, interface_height_m, wavelength_m=None, *, grid='reduced', pressure_pa=None, gas_optical_depth=None, profile_name='user profile')
```

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

## particula.optics.solar.molecular_column

*function*

```python
def molecular_column(wavelength_m, interface_height_m, pressure_pa, gas_optical_depth, *, source='user profile')
```

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

## particula.optics.solar.profile_gas_optical_depth

*function*

```python
def profile_gas_optical_depth(interface_height_m, pressure_pa, temperature_k, mole_fraction, absorption_cross_section_m2)
```

Calculate vertical gas absorption from a sounding and cross sections.

Gas number density follows the ideal gas law at each interface. The
trapezoidal integral assumes density varies linearly with height between
adjacent interfaces. The caller supplies cross sections appropriate to
the pressure, temperature and spectral resolution of each layer.

**Parameters**

- `interface_height_m` (`ArrayLike`) — Strictly descending interface altitudes [m].
- `pressure_pa` (`ArrayLike`) — Nonnegative interface pressures [Pa], increasing downward.
- `temperature_k` (`ArrayLike`) — Positive temperatures [K] at the same interfaces.
- `mole_fraction` (`ArrayLike`) — Nonnegative species mole fractions [mol/mol], shape (interface, gas). Included fractions sum to at most one.
- `absorption_cross_section_m2` (`ArrayLike`) — Nonnegative layer-effective absorption cross sections [m² molecule⁻¹], shape (layer, wavelength, gas), in the same species order as mole_fraction.

**Returns**

- (`ndarray`) — Dimensionless absorption optical depths, shape (layer, wavelength), summed over the supplied gases in top-first layer order.

**Raises**

- `ValueError` — Coordinates, array shapes or physical values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L339-L408)

## particula.optics.solar.solar_irradiance

*function*

```python
def solar_irradiance(wavelength_m)
```

Sample the bundled TSIS-1 extraterrestrial solar spectrum at 1 au.

**Parameters**

- `wavelength_m` (`ArrayLike`) — Increasing vacuum wavelengths [m] within 280–4000 nm.

**Returns**

- (`ndarray`) — Spectral direct-normal irradiance density [W m⁻² m⁻¹], with one value per wavelength, linearly interpolated from the 1 nm reference bins.

**Raises**

- `ValueError` — The wavelength grid is invalid or exceeds the reference.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/solar.py#L215-L234)

## particula.optics.solar.solar_position

*function*

```python
def solar_position(latitude_deg, longitude_deg, time: datetime)
```

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
