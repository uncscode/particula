# particula.coupling.prescribed

*module*

One-way, terrain-following air and settling-particle trajectories.

Map upstream streamline height eta to z = h(x) + J(x) eta with
J = 1 - h/lid. Conserving horizontal volume flux gives u = U(eta)/J.
The fixed lid and Gaussian ridge bound a kinematic, attached flow.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L1-L1)

## particula.coupling.prescribed.SOUNDING_COLUMNS

*attribute* · *module attribute*

```python
SOUNDING_COLUMNS = ('height_m', 'temperature_K', 'pressure_Pa', 'vapor_kg_per_kg_dry_air', 'east_m_s', 'north_m_s')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L16-L23)

## particula.coupling.prescribed.DuneFlow

*class* · *dataclass*

```python
class DuneFlow
```

Gaussian ridge centered at x=0 and uniform along y, in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L91-L240)

### particula.coupling.prescribed.DuneFlow.height

*attribute* · *instance attribute*

```python
height: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L96-L96)

### particula.coupling.prescribed.DuneFlow.lid

*attribute* · *instance attribute*

```python
lid: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L98-L98)

### particula.coupling.prescribed.DuneFlow.sounding

*attribute* · *instance attribute*

```python
sounding: Sounding
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L95-L95)

### particula.coupling.prescribed.DuneFlow.width

*attribute* · *instance attribute*

```python
width: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L97-L97)

### particula.coupling.prescribed.DuneFlow.__init__

*method*

```python
def __init__(sounding: Sounding, height: float, width: float, lid: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L0-L0)

### particula.coupling.prescribed.DuneFlow.__post_init__

*method*

```python
def __post_init__()
```

Require an open passage and sounding coverage of the layer.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L100-L108)

### particula.coupling.prescribed.DuneFlow._wind

*method*

```python
def _wind(x, eta)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L114-L118)

### particula.coupling.prescribed.DuneFlow.ground

*method*

```python
def ground(x)
```

Return bed elevation above the sounding datum in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L110-L112)

### particula.coupling.prescribed.DuneFlow.parcel_trace

*method*

```python
def parcel_trace(parcel_id, xyz, time, aerosol: ParcelInitialState, *, provenance)
```

Return a moving center and replay trace with fixed dry-air mass.

The sounding supplies initial and subsequent thermodynamics. The
aerosol template supplies dry mass, represented counts, radii and κ.
Volume follows the moist gas law. Replay owns aerosol water exchange.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L190-L204)

### particula.coupling.prescribed.DuneFlow.pickup

*method*

```python
def pickup(x, *, reference_height, drag_coefficient, threshold, available_mass, pulse_mass)
```

Return lifted mass (kg) from a finite dry bed patch in one pulse.

Diagnose friction velocity as sqrt(Cd) times the horizontal wind at
reference_height (m above bed). Above threshold (m/s), release the
requested pulse capped by available mass (both kg). The caller debits
the bed once. Cd, threshold and pulse size are experiment inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L206-L240)

### particula.coupling.prescribed.DuneFlow.track

*method*

```python
def track(xyz, time, *, settling=0.0)
```

Advect one center or particle with constant downward slip (m/s).

xyz is a world position in metres and time contains increasing sample
seconds. After first impact a settling particle stays at its impact
point. A zero-slip air marker follows its initial streamline.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L130-L188)

### particula.coupling.prescribed.DuneFlow.velocity

*method*

```python
def velocity(x, z)
```

Sample east, north and vertical wind (m/s) at one world position.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L120-L128)

## particula.coupling.prescribed.Sounding

*class* · *dataclass*

```python
class Sounding
```

Height above a common datum (m), T (K), p (Pa), vapor (kg/kg dry air)
and east/north wind components (m/s) at each level.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L26-L79)

### particula.coupling.prescribed.Sounding.east

*attribute* · *instance attribute*

```python
east: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L36-L36)

### particula.coupling.prescribed.Sounding.height

*attribute* · *instance attribute*

```python
height: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L32-L32)

### particula.coupling.prescribed.Sounding.north

*attribute* · *instance attribute*

```python
north: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L37-L37)

### particula.coupling.prescribed.Sounding.pressure

*attribute* · *instance attribute*

```python
pressure: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L34-L34)

### particula.coupling.prescribed.Sounding.temperature

*attribute* · *instance attribute*

```python
temperature: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L33-L33)

### particula.coupling.prescribed.Sounding.vapor

*attribute* · *instance attribute*

```python
vapor: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L35-L35)

### particula.coupling.prescribed.Sounding.__init__

*method*

```python
def __init__(height: np.ndarray, temperature: np.ndarray, pressure: np.ndarray, vapor: np.ndarray, east: np.ndarray, north: np.ndarray) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L0-L0)

### particula.coupling.prescribed.Sounding.__post_init__

*method*

```python
def __post_init__()
```

Own a finite sorted column and reject invalid thermodynamics.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L39-L58)

### particula.coupling.prescribed.Sounding.from_csv

*method* · *classmethod*

```python
def from_csv(path)
```

Read a normalized radiosonde table with named SI columns.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L60-L66)

### particula.coupling.prescribed.Sounding.thermodynamics

*method*

```python
def thermodynamics(z)
```

Interpolate T, p and vapor at geometric heights in metres.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L68-L79)

## particula.coupling.prescribed.Track

*class* · *dataclass*

```python
class Track
```

World positions (m), sample times (s), and optional impact time (s).

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L82-L88)

### particula.coupling.prescribed.Track.deposition_time

*attribute* · *instance attribute*

```python
deposition_time: float | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L88-L88)

### particula.coupling.prescribed.Track.time

*attribute* · *instance attribute*

```python
time: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L86-L86)

### particula.coupling.prescribed.Track.xyz

*attribute* · *instance attribute*

```python
xyz: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L87-L87)

### particula.coupling.prescribed.Track.__init__

*method*

```python
def __init__(time: np.ndarray, xyz: np.ndarray, deposition_time: float | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L0-L0)

## particula.coupling.prescribed.make_parcel_trace

*function*

```python
def make_parcel_trace(parcel_id, track, aerosol, provenance, temperature, pressure, vapor)
```

Bind sampled SI thermodynamics and fixed dry air to native replay.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/prescribed.py#L243-L267)
