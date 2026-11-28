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

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L21-L28)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L65-L159)

### particula.execution.prepared_recording.PreparedObserver._metadata

*attribute* · *instance attribute*

```python
_metadata = probe.metadata
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L108-L108)

### particula.execution.prepared_recording.PreparedObserver._writer

*attribute* · *instance attribute*

```python
_writer = recorder if isinstance(recorder, AnalysisWriter) else None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L78-L80)

### particula.execution.prepared_recording.PreparedObserver.initial_snapshot_downloads

*attribute* · *instance attribute*

```python
initial_snapshot_downloads = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L115-L115)

### particula.execution.prepared_recording.PreparedObserver.observation_downloads

*attribute* · *instance attribute*

```python
observation_downloads = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L99-L99)

### particula.execution.prepared_recording.PreparedObserver.__init__

*method*

```python
def __init__(simulation, recorder, *, download)
```

Validate the observation profile before the initial full download.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L76-L115)

### particula.execution.prepared_recording.PreparedObserver._flush

*method*

```python
def _flush()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L157-L159)

### particula.execution.prepared_recording.PreparedObserver.record

*method*

```python
def record()
```

Synchronize and append one complete-step observation, if new.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L117-L155)

## particula.execution.prepared_recording._index

*function*

```python
def _index(simulation, time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L52-L62)

## particula.execution.prepared_recording._ready

*function*

```python
def _ready(simulation)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L31-L40)

## particula.execution.prepared_recording._time

*function*

```python
def _time(simulation)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L43-L49)

## particula.execution.prepared_recording.run_prepared

*function*

```python
def run_prepared(simulation, end_time, *, output_times=(), observer=None, optics=None, optical_output_times=None, optical_initial=False, ccn=None, ccn_output_times=None, ccn_initial=False)
```

Run complete fixed steps and explicitly observe the requested grid.

All requested/final times must be on the frozen grid before any work is
enqueued. Older requests are skipped on continuation. Output labels use
initial_time + integer_step * dt, including the final row; decimal inputs
within floating-point roundoff of that grid are admitted. No interpolation,
shortened step, automatic capture or checkpoint is performed. The final
synchronize surfaces asynchronous failures even without an observer.
An OpticalObserver uses its own schedule. By default it records every
accepted step, plus the starting state if optical_initial is true.
CCNObserver supports the same independent scheduling via ccn_output_times.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_recording.py#L162-L249)
