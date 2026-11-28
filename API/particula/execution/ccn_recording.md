# particula.execution.ccn_recording

*module*

Accepted-step equilibrium CCN histories for CPU and prepared Warp owners.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L1-L1)

## particula.execution.ccn_recording.CCNObserver

*class*

```python
class CCNObserver
```

Sample equilibrium CCN on the owner's backend, retaining bounded history.

Common-κ configurations also record critical dry diameters. Slots with
different compositions can have different thresholds. Their bulk output
retains number concentration; evaluate_ccn exposes individual thresholds.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L23-L195)

### particula.execution.ccn_recording.CCNObserver._capacity

*attribute* · *instance attribute*

```python
_capacity = min(self._capacity, self._max_bytes // self._row_bytes)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L92-L92)

### particula.execution.ccn_recording.CCNObserver._device_batch

*attribute* · *instance attribute*

```python
_device_batch = DeviceCCNBatch(simulation, config, self._capacity)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L104-L106)

### particula.execution.ccn_recording.CCNObserver._fixed

*attribute* · *instance attribute*

```python
_fixed = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L68-L68)

### particula.execution.ccn_recording.CCNObserver._max_bytes

*attribute* · *instance attribute*

```python
_max_bytes = index(max_bytes)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L45-L45)

### particula.execution.ccn_recording.CCNObserver._metadata

*attribute* · *instance attribute*

```python
_metadata = {'schema': 'equilibrium-ccn-v1', 'configuration': asdict(config), 'distribution_type': distribution, 'boxes': boxes, 'materials': materials, 'axes': ['time', 'box', 'saturation_ratio'], 'units': _history_units(config)}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L77-L85)

### particula.execution.ccn_recording.CCNObserver._pending

*attribute* · *instance attribute*

```python
_pending = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L97-L97)

### particula.execution.ccn_recording.CCNObserver._published

*attribute* · *instance attribute*

```python
_published = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L96-L96)

### particula.execution.ccn_recording.CCNObserver._row_bytes

*attribute* · *instance attribute*

```python
_row_bytes = 8 * 1 + boxes * len(config.saturation_ratios) * 1 + int(config.kappa is not None)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L86-L91)

### particula.execution.ccn_recording.CCNObserver._size

*attribute* · *instance attribute*

```python
_size = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L98-L98)

### particula.execution.ccn_recording.CCNObserver._times

*attribute* · *instance attribute*

```python
_times = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L95-L95)

### particula.execution.ccn_recording.CCNObserver.config

*attribute* · *instance attribute*

```python
config = config
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L41-L41)

### particula.execution.ccn_recording.CCNObserver.simulation

*attribute* · *instance attribute*

```python
simulation = simulation
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L42-L42)

### particula.execution.ccn_recording.CCNObserver.__init__

*method*

```python
def __init__(simulation, config, *, batch_size=32, max_bytes=64 << 20)
```

Bind a frozen configuration and reserve capacity for compact rows.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L31-L106)

### particula.execution.ccn_recording.CCNObserver._check_start

*method*

```python
def _check_start(time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L108-L116)

### particula.execution.ccn_recording.CCNObserver._record

*method*

```python
def _record(time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L129-L151)

### particula.execution.ccn_recording.CCNObserver.arrays

*method*

```python
def arrays()
```

Flush and return detached arrays on their recorded time grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L171-L179)

### particula.execution.ccn_recording.CCNObserver.flush

*method*

```python
def flush()
```

Transfer and publish the complete pending batch after validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L153-L169)

### particula.execution.ccn_recording.CCNObserver.latest

*method*

```python
def latest()
```

Flush and return only the latest detached observation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L181-L189)

### particula.execution.ccn_recording.CCNObserver.record

*method*

```python
def record()
```

Sample one accepted state, skipping the equal-time continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L118-L127)

### particula.execution.ccn_recording.CCNObserver.save

*method*

```python
def save(path)
```

Save compact rows and configuration by atomic replacement.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L191-L195)

## particula.execution.ccn_recording._history_units

*function*

```python
def _history_units(config)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L16-L20)

## particula.execution.ccn_recording.read_ccn_history

*function*

```python
def read_ccn_history(path, *, max_bytes=128 << 20)
```

Read CCN output with units, axes and material configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ccn_recording.py#L198-L234)
