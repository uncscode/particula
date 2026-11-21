# particula.execution.optical_recording

*module*

Bounded optical histories independent of aerosol snapshots.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L1-L1)

## particula.execution.optical_recording._CHANNELS

*attribute* · *module attribute*

```python
_CHANNELS = PopulationOptics._fields
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L15-L15)

## particula.execution.optical_recording._DEFAULT

*attribute* · *module attribute*

```python
_DEFAULT = ('extinction', 'scattering', 'absorption', 'single_scattering_albedo', 'asymmetry', 'extinction_valid', 'scattering_valid')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L16-L24)

## particula.execution.optical_recording.OpticalHistory

*class*

```python
class OpticalHistory
```

Append and read atomic optical chunks in a SQLite file.

Every chunk has time [s] and bulk arrays with axes (time, box, wavelength).
Coefficients are in inverse metres. Albedo and asymmetry are dimensionless
and accompanied by validity masks. One writer owns the observation buffer.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L46-L188)

### particula.execution.optical_recording.OpticalHistory._last_time

*attribute* · *instance attribute*

```python
_last_time = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L77-L77)

### particula.execution.optical_recording.OpticalHistory._provenance

*attribute* · *instance attribute*

```python
_provenance = json.dumps(provenance or {}, sort_keys=True, allow_nan=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L74-L76)

### particula.execution.optical_recording.OpticalHistory.config

*attribute* · *instance attribute*

```python
config = config
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L72-L72)

### particula.execution.optical_recording.OpticalHistory.path

*attribute* · *instance attribute*

```python
path = Path(path)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L71-L71)

### particula.execution.optical_recording.OpticalHistory.variables

*attribute* · *instance attribute*

```python
variables = _channels(variables)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L73-L73)

### particula.execution.optical_recording.OpticalHistory.water_vapor

*attribute* · *instance attribute*

```python
water_vapor = water_vapor
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L70-L70)

### particula.execution.optical_recording.OpticalHistory.last_time

*property*

```python
last_time
```

Time of the last committed row, or None for an empty history.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L80-L82)

### particula.execution.optical_recording.OpticalHistory.__init__

*method*

```python
def __init__(path, config, *, variables=_DEFAULT, provenance=None, water_vapor=None)
```

Configure output; a named water gas enables box RH observations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L54-L77)

### particula.execution.optical_recording.OpticalHistory._bind

*method*

```python
def _bind(boxes, distribution)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L93-L148)

### particula.execution.optical_recording.OpticalHistory._water_lane

*method*

```python
def _water_lane(gas_names)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L84-L91)

### particula.execution.optical_recording.OpticalHistory._write

*method*

```python
def _write(times, values, humidity)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L150-L188)

## particula.execution.optical_recording.OpticalObserver

*class*

```python
class OpticalObserver
```

Bind a bulk optical history to a CPU or fixed-step Warp owner.

Use run_until or run_prepared with optics=observer, or call record()
at chosen accepted states and flush() before closing the owner.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L231-L355)

### particula.execution.optical_recording.OpticalObserver._batch_size

*attribute* · *instance attribute*

```python
_batch_size = index(batch_size)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L260-L260)

### particula.execution.optical_recording.OpticalObserver._cpu_humidity

*attribute* · *instance attribute*

```python
_cpu_humidity = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L265-L265)

### particula.execution.optical_recording.OpticalObserver._cpu_rows

*attribute* · *instance attribute*

```python
_cpu_rows = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L264-L264)

### particula.execution.optical_recording.OpticalObserver._device_batch

*attribute* · *instance attribute*

```python
_device_batch = DeviceOpticalBatch(simulation, history.config, self._batch_size, particle_tile, coefficient_capacity, water_lane=water_lane)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L285-L292)

### particula.execution.optical_recording.OpticalObserver._times

*attribute* · *instance attribute*

```python
_times = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L263-L263)

### particula.execution.optical_recording.OpticalObserver._water_lane

*attribute* · *instance attribute*

```python
_water_lane = history._water_lane(simulation._state.gas.name)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L269-L269)

### particula.execution.optical_recording.OpticalObserver.history

*attribute* · *instance attribute*

```python
history = history
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L259-L259)

### particula.execution.optical_recording.OpticalObserver.simulation

*attribute* · *instance attribute*

```python
simulation = simulation
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L258-L258)

### particula.execution.optical_recording.OpticalObserver.__init__

*method*

```python
def __init__(simulation, history, *, batch_size=32, particle_tile=256, coefficient_capacity=10090)
```

Allocate bounded observation storage for this owner.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L238-L297)

### particula.execution.optical_recording.OpticalObserver._check_start

*method*

```python
def _check_start(time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L299-L303)

### particula.execution.optical_recording.OpticalObserver._publish

*method*

```python
def _publish(values, humidity)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L351-L355)

### particula.execution.optical_recording.OpticalObserver._record

*method*

```python
def _record(time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L316-L336)

### particula.execution.optical_recording.OpticalObserver.flush

*method*

```python
def flush()
```

Publish all buffered rows once every row has passed validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L338-L349)

### particula.execution.optical_recording.OpticalObserver.record

*method*

```python
def record()
```

Evaluate live state and skip duplicate continuation times.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L305-L314)

## particula.execution.optical_recording._channels

*function*

```python
def _channels(variables)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L27-L43)

## particula.execution.optical_recording.read_optical_history

*function*

```python
def read_optical_history(path)
```

Yield metadata and arrays for each committed chunk.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/optical_recording.py#L191-L228)
