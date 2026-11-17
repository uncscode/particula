# particula.execution.prepared_recording

*module*

Explicit prepared-GPU observation transfers on the frozen step grid.

Attaching an observer downloads one native snapshot to bind analysis metadata.
Subsequent observations explicitly choose reduced diagnostics or full snapshots.
The runner never changes the prepared timestep, capture plan or RNG schedule.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L1-L1)

## particula.execution.prepared_recording._REDUCED

*attribute* · *module attribute*

```python
_REDUCED = {'number_density': 'number_concentration', 'particle_mass_density': 'particle_mass_concentration', 'gas_concentration': 'gas_concentration', 'temperature': 'temperature', 'pressure': 'pressure', 'volume': 'volume'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L15-L22)

## particula.execution.prepared_recording.PreparedObserver

*class*

```python
class PreparedObserver
```

Bind a Recorder to one prepared owner and an explicit download policy.

Construction downloads one full native snapshot for immutable species,
dimensions and measure metadata, but appends no row. download='diagnostics'
subsequently downloads only existing per-box/species reductions and forcing;
slot variables require download='snapshot'. Returned history is detached.
Existing captured *replay* may be observed between steps; running this host
workflow inside an active CUDA capture is rejected before any transfer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L59-L153)

### particula.execution.prepared_recording.PreparedObserver._metadata

*attribute* · *instance attribute*

```python
_metadata = probe.metadata
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L102-L102)

### particula.execution.prepared_recording.PreparedObserver._writer

*attribute* · *instance attribute*

```python
_writer = recorder if isinstance(recorder, AnalysisWriter) else None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L72-L74)

### particula.execution.prepared_recording.PreparedObserver.initial_snapshot_downloads

*attribute* · *instance attribute*

```python
initial_snapshot_downloads = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L109-L109)

### particula.execution.prepared_recording.PreparedObserver.observation_downloads

*attribute* · *instance attribute*

```python
observation_downloads = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L93-L93)

### particula.execution.prepared_recording.PreparedObserver.__init__

*method*

```python
def __init__(simulation, recorder, *, download)
```

Validate the observation profile before the initial full download.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L70-L109)

### particula.execution.prepared_recording.PreparedObserver._flush

*method*

```python
def _flush()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L151-L153)

### particula.execution.prepared_recording.PreparedObserver.record

*method*

```python
def record()
```

Synchronize and append one complete-step observation, if new.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L111-L149)

## particula.execution.prepared_recording._index

*function*

```python
def _index(simulation, time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L46-L56)

## particula.execution.prepared_recording._ready

*function*

```python
def _ready(simulation)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L25-L34)

## particula.execution.prepared_recording._time

*function*

```python
def _time(simulation)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L37-L43)

## particula.execution.prepared_recording.run_prepared

*function*

```python
def run_prepared(simulation, end_time, *, output_times=(), observer=None)
```

Run complete fixed steps and explicitly observe the requested grid.

All requested/final times must be on the frozen grid before any work is
enqueued. Older requests are skipped on continuation. Output labels use
initial_time + integer_step * dt, including the final row; decimal inputs
within floating-point roundoff of that grid are admitted. No interpolation,
shortened step, automatic capture or checkpoint is performed. The final
synchronize surfaces asynchronous failures even without an observer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L156-L204)
