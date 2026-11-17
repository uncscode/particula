# particula.execution.recording

*module*

Detached native observations at accepted complete-step physical times.

Recording has no access to process RNG or intermediate trial states. Requested
output times explicitly shorten outer steps; this may alter a numerical or
stochastic trajectory. An unchanged step schedule is required for bitwise
observer-invariance claims. Full particle arrays are opt-in and byte bounded.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L1-L1)

## particula.execution.recording.OBSERVABLES

*attribute* · *module attribute*

```python
OBSERVABLES = {'number_density': ('particles.number_density', 'm^-3', ('box')), 'particle_mass_density': ('particles.species_mass_density', 'kg m^-3', ('box', 'material')), 'gas_concentration': ('gas.concentration', 'kg m^-3', ('box', 'gas')), 'temperature': ('environment.temperature', 'K', ('box')), 'pressure': ('environment.pressure', 'Pa', ('box')), 'volume': ('particles.volume', 'm^3', ('box')), 'particle_masses': ('particles.masses', 'kg', ('box', 'slot', 'material')), 'particle_concentration': ('particles.concentration', 'population_measure', ('box', 'slot')), 'particle_charge': ('particles.charge', 'elementary_charge', ('box', 'slot'))}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L16-L38)

## particula.execution.recording.Recorder

*class*

```python
class Recorder
```

Collect selected detached observations with explicit storage limits.

metadata retains full gas order, distribution interpretation, material
densities and dimensions. It is analysis provenance, not a restart state.
max_bytes bounds retained numeric observations including times; Python
object overhead is additional. Full slot arrays are opt-in. Volume is always
recorded so population measures remain interpretable.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L41-L191)

### particula.execution.recording.Recorder._bytes

*attribute* · *instance attribute*

```python
_bytes = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L83-L83)

### particula.execution.recording.Recorder._metadata

*attribute* · *instance attribute*

```python
_metadata = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L82-L82)

### particula.execution.recording.Recorder._rows

*attribute* · *instance attribute*

```python
_rows: dict[str, list[np.ndarray]] = {name: [] for name in self.variables}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L79-L81)

### particula.execution.recording.Recorder._times

*attribute* · *instance attribute*

```python
_times: list[float] = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L78-L78)

### particula.execution.recording.Recorder.variables

*attribute* · *instance attribute*

```python
variables = tuple(variables)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L63-L63)

### particula.execution.recording.Recorder.last_time

*property*

```python
last_time
```

Return the last observation time or None for an empty history.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L91-L93)

### particula.execution.recording.Recorder.metadata

*property*

```python
metadata
```

Return detached dimensions, units and species provenance.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L86-L88)

### particula.execution.recording.Recorder.__init__

*method*

```python
def __init__(variables=('number_density', 'particle_mass_density', 'gas_concentration'), *, max_records=100000, max_bytes=64 * 1024 * 1024)
```

Validate names and limits without accessing simulation state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L51-L83)

### particula.execution.recording.Recorder._append

*method*

```python
def _append(time, metadata, row)
```

Append native/reduced values through one atomic boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L153-L182)

### particula.execution.recording.Recorder.arrays

*method*

```python
def arrays()
```

Return detached stacked arrays; time always has units of seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L184-L191)

### particula.execution.recording.Recorder.record

*method*

```python
def record(time, state)
```

Append one accepted native observation; equal-time resume is a no-op.

All metadata, shapes and storage limits validate before append. Layout
changes require a new recorder; widths do not imply species order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L95-L151)

## particula.execution.recording.run_until

*function*

```python
def run_until(simulation, end_time, *, output_times=(), recorder=None)
```

Advance an owner and observe requested times plus final time.

The owner supplies time, step(bound), and detached snapshot(). Every output
is at a complete accepted step. Resumed calls skip older output requests and
Recorder prevents duplicates at the joining time. Adaptive GPU observation
explicitly downloads a full accepted snapshot at each requested output.
Fixed prepared owners instead use run_prepared and PreparedObserver, which
define their grid admission and explicit reduced/full download profiles.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/recording.py#L194-L235)
