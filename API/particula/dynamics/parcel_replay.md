# particula.dynamics.parcel_replay

*module*

Replay parcel forcing and checkpoint extensive species inventories.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L1-L1)

## particula.dynamics.parcel_replay.OrganicForcing

*class* · *dataclass*

```python
class OrganicForcing
```

Re-exported from: `particula.dynamics.parcel_trace`

Dilute tracer uptake with a prescribed first-order rate (1/s).

Gas and particulate inventories are in kg. Rates (kg/s) and totals (kg)
are keyed to successive trace intervals. Organic uptake is a passive
tracer experiment and does not change the water-equilibrium curve.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L45-L76)

### particula.dynamics.parcel_replay.OrganicForcing.initial_gas

*attribute* · *class attribute* · *instance attribute*

```python
initial_gas: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L55-L55)

### particula.dynamics.parcel_replay.OrganicForcing.name

*attribute* · *class attribute* · *instance attribute*

```python
name: str = 'organic'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L54-L54)

### particula.dynamics.parcel_replay.OrganicForcing.rates

*attribute* · *class attribute* · *instance attribute*

```python
rates: tuple[float, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L57-L57)

### particula.dynamics.parcel_replay.OrganicForcing.totals

*attribute* · *class attribute* · *instance attribute*

```python
totals: tuple[float, ...] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L58-L58)

### particula.dynamics.parcel_replay.OrganicForcing.uptake_rate

*attribute* · *class attribute* · *instance attribute*

```python
uptake_rate: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L56-L56)

### particula.dynamics.parcel_replay.OrganicForcing.__init__

*method*

```python
def __init__(name: str = 'organic', initial_gas: float = 0.0, uptake_rate: float = 0.0, rates: tuple[float, ...] = (), totals: tuple[float, ...] = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L0-L0)

### particula.dynamics.parcel_replay.OrganicForcing.__post_init__

*method*

```python
def __post_init__()
```

Validate the physical inputs and retain owned values.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L60-L76)

## particula.dynamics.parcel_replay.ParcelTrace

*class* · *dataclass*

```python
class ParcelTrace
```

Re-exported from: `particula.dynamics.parcel_trace`

One forcing history with optional wet-radius and liquid references.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L79-L359)

### particula.dynamics.parcel_replay.ParcelTrace.dry_density

*attribute* · *class attribute* · *instance attribute*

```python
dry_density: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L96-L96)

### particula.dynamics.parcel_replay.ParcelTrace.height

*attribute* · *instance attribute*

```python
height: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L86-L86)

### particula.dynamics.parcel_replay.ParcelTrace.initial

*attribute* · *instance attribute*

```python
initial: ParcelInitialState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L91-L91)

### particula.dynamics.parcel_replay.ParcelTrace.initial_wet

*attribute* · *instance attribute*

```python
initial_wet: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L92-L92)

### particula.dynamics.parcel_replay.ParcelTrace.organic

*attribute* · *class attribute* · *instance attribute*

```python
organic: OrganicForcing = OrganicForcing()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L93-L93)

### particula.dynamics.parcel_replay.ParcelTrace.parcel_id

*attribute* · *instance attribute*

```python
parcel_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L83-L83)

### particula.dynamics.parcel_replay.ParcelTrace.pressure

*attribute* · *instance attribute*

```python
pressure: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L88-L88)

### particula.dynamics.parcel_replay.ParcelTrace.provenance

*attribute* · *instance attribute*

```python
provenance: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L84-L84)

### particula.dynamics.parcel_replay.ParcelTrace.reference_liquid

*attribute* · *class attribute* · *instance attribute*

```python
reference_liquid: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L95-L95)

### particula.dynamics.parcel_replay.ParcelTrace.reference_radius

*attribute* · *class attribute* · *instance attribute*

```python
reference_radius: np.ndarray | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L94-L94)

### particula.dynamics.parcel_replay.ParcelTrace.temperature

*attribute* · *instance attribute*

```python
temperature: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L87-L87)

### particula.dynamics.parcel_replay.ParcelTrace.time

*attribute* · *instance attribute*

```python
time: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L85-L85)

### particula.dynamics.parcel_replay.ParcelTrace.vapor

*attribute* · *instance attribute*

```python
vapor: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L90-L90)

### particula.dynamics.parcel_replay.ParcelTrace.volume

*attribute* · *instance attribute*

```python
volume: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L89-L89)

### particula.dynamics.parcel_replay.ParcelTrace.__init__

*method*

```python
def __init__(parcel_id: str, provenance: str, time: np.ndarray, height: np.ndarray, temperature: np.ndarray, pressure: np.ndarray, volume: np.ndarray, vapor: np.ndarray, initial: ParcelInitialState, initial_wet: np.ndarray, organic: OrganicForcing = OrganicForcing(), reference_radius: np.ndarray | None = None, reference_liquid: np.ndarray | None = None, dry_density: float | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L0-L0)

### particula.dynamics.parcel_replay.ParcelTrace.__post_init__

*method*

```python
def __post_init__()
```

Validate the physical inputs and retain owned values.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L98-L138)

### particula.dynamics.parcel_replay.ParcelTrace._metadata

*method*

```python
def _metadata()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L253-L266)

### particula.dynamics.parcel_replay.ParcelTrace._validate_initial_state

*method*

```python
def _validate_initial_state()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L157-L197)

### particula.dynamics.parcel_replay.ParcelTrace._validate_reference_outputs

*method*

```python
def _validate_reference_outputs()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L140-L155)

### particula.dynamics.parcel_replay.ParcelTrace.identity

*method*

```python
def identity()
```

Digest of forcing, owners and initial state for continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L268-L282)

### particula.dynamics.parcel_replay.ParcelTrace.load

*method* · *classmethod*

```python
def load(path)
```

Read one trajectory and verify its declared units and owners.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L318-L359)

### particula.dynamics.parcel_replay.ParcelTrace.record

*method* · *classmethod*

```python
def record(trajectory, parcel_id='ascent', provenance='Particula-generated ascent')
```

Record actual output times, including the initial wet spectrum.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L199-L221)

### particula.dynamics.parcel_replay.ParcelTrace.sample_every

*method*

```python
def sample_every(stride)
```

Thin forcing records and retain both endpoints.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L223-L251)

### particula.dynamics.parcel_replay.ParcelTrace.save

*method*

```python
def save(path)
```

Write CSV forcing and sibling JSON with metadata and comparisons.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_trace.py#L284-L316)

## particula.dynamics.parcel_replay.ReplayCheckpoint

*class* · *dataclass*

```python
class ReplayCheckpoint
```

Portable, immutable continuation at a completed forcing record.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L20-L47)

### particula.dynamics.parcel_replay.ReplayCheckpoint.humidity_amplitude

*attribute* · *instance attribute*

```python
humidity_amplitude: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L30-L30)

### particula.dynamics.parcel_replay.ReplayCheckpoint.index

*attribute* · *instance attribute*

```python
index: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L27-L27)

### particula.dynamics.parcel_replay.ReplayCheckpoint.inventory

*attribute* · *instance attribute*

```python
inventory: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L31-L31)

### particula.dynamics.parcel_replay.ReplayCheckpoint.max_step

*attribute* · *instance attribute*

```python
max_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L29-L29)

### particula.dynamics.parcel_replay.ReplayCheckpoint.parcel_id

*attribute* · *instance attribute*

```python
parcel_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L25-L25)

### particula.dynamics.parcel_replay.ReplayCheckpoint.species

*attribute* · *instance attribute*

```python
species: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L26-L26)

### particula.dynamics.parcel_replay.ReplayCheckpoint.time

*attribute* · *instance attribute*

```python
time: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L28-L28)

### particula.dynamics.parcel_replay.ReplayCheckpoint.trace_id

*attribute* · *instance attribute*

```python
trace_id: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L24-L24)

### particula.dynamics.parcel_replay.ReplayCheckpoint.__init__

*method*

```python
def __init__(trace_id: str, parcel_id: str, species: tuple, index: int, time: float, max_step: float, humidity_amplitude: float, inventory: tuple) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L0-L0)

### particula.dynamics.parcel_replay.ReplayCheckpoint.load

*method* · *classmethod*

```python
def load(path)
```

Read a checkpoint for validation at continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L37-L47)

### particula.dynamics.parcel_replay.ReplayCheckpoint.save

*method*

```python
def save(path)
```

Write the completed checkpoint to JSON.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L33-L35)

## particula.dynamics.parcel_replay.ReplayResult

*class* · *dataclass*

```python
class ReplayResult
```

Record-time outputs. Water and organic budgets are extensive kg.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L50-L71)

### particula.dynamics.parcel_replay.ReplayResult.bin_organic

*attribute* · *instance attribute*

```python
bin_organic: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L66-L66)

### particula.dynamics.parcel_replay.ReplayResult.bin_water

*attribute* · *instance attribute*

```python
bin_water: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L65-L65)

### particula.dynamics.parcel_replay.ReplayResult.checkpoint

*attribute* · *instance attribute*

```python
checkpoint: ReplayCheckpoint | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L71-L71)

### particula.dynamics.parcel_replay.ReplayResult.dry_mass

*attribute* · *instance attribute*

```python
dry_mass: np.ndarray | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L69-L69)

### particula.dynamics.parcel_replay.ReplayResult.gas_masses

*attribute* · *instance attribute*

```python
gas_masses: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L68-L68)

### particula.dynamics.parcel_replay.ReplayResult.gas_species

*attribute* · *instance attribute*

```python
gas_species: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L67-L67)

### particula.dynamics.parcel_replay.ReplayResult.humidity_gradient

*attribute* · *instance attribute*

```python
humidity_gradient: float | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L70-L70)

### particula.dynamics.parcel_replay.ReplayResult.liquid

*attribute* · *instance attribute*

```python
liquid: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L57-L57)

### particula.dynamics.parcel_replay.ReplayResult.number

*attribute* · *instance attribute*

```python
number: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L64-L64)

### particula.dynamics.parcel_replay.ReplayResult.organic_added

*attribute* · *instance attribute*

```python
organic_added: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L63-L63)

### particula.dynamics.parcel_replay.ReplayResult.organic_gas

*attribute* · *instance attribute*

```python
organic_gas: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L61-L61)

### particula.dynamics.parcel_replay.ReplayResult.organic_particle

*attribute* · *instance attribute*

```python
organic_particle: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L62-L62)

### particula.dynamics.parcel_replay.ReplayResult.supersaturation

*attribute* · *instance attribute*

```python
supersaturation: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L58-L58)

### particula.dynamics.parcel_replay.ReplayResult.time

*attribute* · *instance attribute*

```python
time: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L54-L54)

### particula.dynamics.parcel_replay.ReplayResult.volume

*attribute* · *instance attribute*

```python
volume: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L55-L55)

### particula.dynamics.parcel_replay.ReplayResult.water_exchange

*attribute* · *instance attribute*

```python
water_exchange: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L59-L59)

### particula.dynamics.parcel_replay.ReplayResult.water_restored

*attribute* · *instance attribute*

```python
water_restored: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L60-L60)

### particula.dynamics.parcel_replay.ReplayResult.wet_radius

*attribute* · *instance attribute*

```python
wet_radius: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L56-L56)

### particula.dynamics.parcel_replay.ReplayResult.__init__

*method*

```python
def __init__(time: np.ndarray, volume: np.ndarray, wet_radius: np.ndarray, liquid: np.ndarray, supersaturation: np.ndarray, water_exchange: np.ndarray, water_restored: np.ndarray, organic_gas: np.ndarray, organic_particle: np.ndarray, organic_added: np.ndarray, number: np.ndarray, bin_water: np.ndarray, bin_organic: np.ndarray, gas_species: tuple[str, ...], gas_masses: np.ndarray, dry_mass: np.ndarray | None, humidity_gradient: float | None, checkpoint: ReplayCheckpoint | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L0-L0)

## particula.dynamics.parcel_replay.replay

*function*

```python
def replay(trace, *, max_step, device='cpu', humidity_amplitude=0.0, gradient=False, species=None, stop_index=None, checkpoint=None)
```

Integrate prescribed humid air and a first-order organic uptake tracer.

The humidity perturbation multiplies vapor by 1 + amplitude × normalized
elapsed time. A reverse pass gives d(terminal liquid kg)/d(amplitude).
Sources use interval rates and one impulse at the right endpoint.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel_replay.py#L74-L109)
