# particula.dynamics.parcel_trace

*module*

Offline warm-parcel forcing with SI units and extensive inventories.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L1-L1)

## particula.dynamics.parcel_trace.COLUMNS

*attribute* · *module attribute*

```python
COLUMNS = ('parcel_id', 'time_s', 'height_m', 'temperature_K', 'pressure_Pa', 'volume_m3', 'water_kg_per_kg_dry_air')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L16-L24)

## particula.dynamics.parcel_trace.CONVENTIONS

*attribute* · *module attribute*

```python
CONVENTIONS = {'schema': 1, 'columns': list(COLUMNS), 'humidity': 'prescribed water vapor mass per dry-air mass', 'sampling': 'linear state interpolation at recorded times', 'coordinates': 'height in metres, positive upward', 'interval_forcing': 'rates in kg/s throughout interval; totals in kg at its end'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L25-L34)

## particula.dynamics.parcel_trace.OrganicForcing

*class* · *dataclass*

```python
class OrganicForcing
```

Dilute tracer uptake with a prescribed first-order rate (1/s).

Gas and particulate inventories are in kg. Rates (kg/s) and totals (kg)
are keyed to successive trace intervals. Organic uptake is a passive
tracer experiment and does not change the water-equilibrium curve.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L45-L76)

### particula.dynamics.parcel_trace.OrganicForcing.initial_gas

*attribute* · *class attribute* · *instance attribute*

```python
initial_gas: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L55-L55)

### particula.dynamics.parcel_trace.OrganicForcing.name

*attribute* · *class attribute* · *instance attribute*

```python
name: str = 'organic'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L54-L54)

### particula.dynamics.parcel_trace.OrganicForcing.rates

*attribute* · *class attribute* · *instance attribute*

```python
rates: tuple[float, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L57-L57)

### particula.dynamics.parcel_trace.OrganicForcing.totals

*attribute* · *class attribute* · *instance attribute*

```python
totals: tuple[float, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L58-L58)

### particula.dynamics.parcel_trace.OrganicForcing.uptake_rate

*attribute* · *class attribute* · *instance attribute*

```python
uptake_rate: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L56-L56)

### particula.dynamics.parcel_trace.OrganicForcing.__init__

*method*

```python
def __init__(name: str = 'organic', initial_gas: float = 0.0, uptake_rate: float = 0.0, rates: tuple[float, ...] = (), totals: tuple[float, ...] = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L0-L0)

### particula.dynamics.parcel_trace.OrganicForcing.__post_init__

*method*

```python
def __post_init__()
```

Validate the physical inputs and retain owned values.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L60-L76)

## particula.dynamics.parcel_trace.ParcelTrace

*class* · *dataclass*

```python
class ParcelTrace
```

One forcing history with optional wet-radius and liquid references.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L79-L359)

### particula.dynamics.parcel_trace.ParcelTrace.dry_density

*attribute* · *class attribute* · *instance attribute*

```python
dry_density: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L96-L96)

### particula.dynamics.parcel_trace.ParcelTrace.height

*attribute* · *instance attribute*

```python
height: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L86-L86)

### particula.dynamics.parcel_trace.ParcelTrace.initial

*attribute* · *instance attribute*

```python
initial: ParcelInitialState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L91-L91)

### particula.dynamics.parcel_trace.ParcelTrace.initial_wet

*attribute* · *instance attribute*

```python
initial_wet: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L92-L92)

### particula.dynamics.parcel_trace.ParcelTrace.organic

*attribute* · *class attribute* · *instance attribute*

```python
organic: OrganicForcing = OrganicForcing()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L93-L93)

### particula.dynamics.parcel_trace.ParcelTrace.parcel_id

*attribute* · *instance attribute*

```python
parcel_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L83-L83)

### particula.dynamics.parcel_trace.ParcelTrace.pressure

*attribute* · *instance attribute*

```python
pressure: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L88-L88)

### particula.dynamics.parcel_trace.ParcelTrace.provenance

*attribute* · *instance attribute*

```python
provenance: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L84-L84)

### particula.dynamics.parcel_trace.ParcelTrace.reference_liquid

*attribute* · *class attribute* · *instance attribute*

```python
reference_liquid: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L95-L95)

### particula.dynamics.parcel_trace.ParcelTrace.reference_radius

*attribute* · *class attribute* · *instance attribute*

```python
reference_radius: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L94-L94)

### particula.dynamics.parcel_trace.ParcelTrace.temperature

*attribute* · *instance attribute*

```python
temperature: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L87-L87)

### particula.dynamics.parcel_trace.ParcelTrace.time

*attribute* · *instance attribute*

```python
time: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L85-L85)

### particula.dynamics.parcel_trace.ParcelTrace.vapor

*attribute* · *instance attribute*

```python
vapor: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L90-L90)

### particula.dynamics.parcel_trace.ParcelTrace.volume

*attribute* · *instance attribute*

```python
volume: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L89-L89)

### particula.dynamics.parcel_trace.ParcelTrace.__init__

*method*

```python
def __init__(parcel_id: str, provenance: str, time: np.ndarray, height: np.ndarray, temperature: np.ndarray, pressure: np.ndarray, volume: np.ndarray, vapor: np.ndarray, initial: ParcelInitialState, initial_wet: np.ndarray, organic: OrganicForcing = OrganicForcing(), reference_radius: np.ndarray | None = None, reference_liquid: np.ndarray | None = None, dry_density: float | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L0-L0)

### particula.dynamics.parcel_trace.ParcelTrace.__post_init__

*method*

```python
def __post_init__()
```

Validate the physical inputs and retain owned values.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L98-L138)

### particula.dynamics.parcel_trace.ParcelTrace._metadata

*method*

```python
def _metadata()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L253-L266)

### particula.dynamics.parcel_trace.ParcelTrace._validate_initial_state

*method*

```python
def _validate_initial_state()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L157-L197)

### particula.dynamics.parcel_trace.ParcelTrace._validate_reference_outputs

*method*

```python
def _validate_reference_outputs()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L140-L155)

### particula.dynamics.parcel_trace.ParcelTrace.identity

*method*

```python
def identity()
```

Digest of forcing, owners and initial state for continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L268-L282)

### particula.dynamics.parcel_trace.ParcelTrace.load

*method* · *classmethod*

```python
def load(path)
```

Read one trajectory and verify its declared units and owners.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L318-L359)

### particula.dynamics.parcel_trace.ParcelTrace.record

*method* · *classmethod*

```python
def record(trajectory, parcel_id='ascent', provenance='Particula-generated ascent')
```

Record actual output times, including the initial wet spectrum.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L199-L221)

### particula.dynamics.parcel_trace.ParcelTrace.sample_every

*method*

```python
def sample_every(stride)
```

Thin forcing records and retain both endpoints.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L223-L251)

### particula.dynamics.parcel_trace.ParcelTrace.save

*method*

```python
def save(path)
```

Write CSV forcing and sibling JSON with metadata and comparisons.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L284-L316)

## particula.dynamics.parcel_trace._vector

*function*

```python
def _vector(value, name)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L37-L42)
