# particula.dynamics._parcel_experiment

*module*

Common schedule, budgets and device dispatch for parcel experiments.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L1-L1)

## particula.dynamics._parcel_experiment.ClusterResult

*class* · *dataclass*

```python
class ClusterResult
```

Common observations and cumulative donor-to-receiver transfers.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L14-L24)

### particula.dynamics._parcel_experiment.ClusterResult.flow_gradient

*attribute* · *instance attribute*

```python
flow_gradient: float | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L24-L24)

### particula.dynamics._parcel_experiment.ClusterResult.inventories_transferred

*attribute* · *instance attribute*

```python
inventories_transferred: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L21-L21)

### particula.dynamics._parcel_experiment.ClusterResult.parcels

*attribute* · *instance attribute*

```python
parcels: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L19-L19)

### particula.dynamics._parcel_experiment.ClusterResult.schedule

*attribute* · *instance attribute*

```python
schedule: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L23-L23)

### particula.dynamics._parcel_experiment.ClusterResult.time

*attribute* · *instance attribute*

```python
time: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L18-L18)

### particula.dynamics._parcel_experiment.ClusterResult.totals

*attribute* · *instance attribute*

```python
totals: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L22-L22)

### particula.dynamics._parcel_experiment.ClusterResult.transfers

*attribute* · *instance attribute*

```python
transfers: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L20-L20)

### particula.dynamics._parcel_experiment.ClusterResult.__init__

*method*

```python
def __init__(time: np.ndarray, parcels: dict, transfers: dict, inventories_transferred: dict, totals: dict, schedule: np.ndarray, flow_gradient: float | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L0-L0)

## particula.dynamics._parcel_experiment._exchange_schedule

*function*

```python
def _exchange_schedule(ids, connections, steps, volumes)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L140-L168)

## particula.dynamics._parcel_experiment._experiment_times

*function*

```python
def _experiment_times(traces, connections, max_step, amplitude, species, checkpoint, stop_index)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L291-L322)

## particula.dynamics._parcel_experiment._face_ledgers

*function*

```python
def _face_ledgers(fluxes, obs, pairs, edges, dry, traces)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L190-L218)

## particula.dynamics._parcel_experiment._forcing

*function*

```python
def _forcing(trace, steps, amplitude)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L61-L100)

## particula.dynamics._parcel_experiment._initial

*function*

```python
def _initial(traces, bins)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L103-L117)

## particula.dynamics._parcel_experiment._observation

*function*

```python
def _observation(ids, pairs, humidity_target, flow_target)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L171-L187)

## particula.dynamics._parcel_experiment._parcel_result

*function*

```python
def _parcel_result(t, s, times, volume, vapor, species, end, max_step, amplitude, humidity_gradient)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L221-L288)

## particula.dynamics._parcel_experiment._restore

*function*

```python
def _restore(trace, species, checkpoint, max_step, amplitude)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L120-L137)

## particula.dynamics._parcel_experiment._restore_inventory

*function*

```python
def _restore_inventory(checkpoint, initial)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L325-L335)

## particula.dynamics._parcel_experiment._schedule

*function*

```python
def _schedule(traces, start, end, max_step, contacts=())
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L27-L58)

## particula.dynamics._parcel_experiment.integrate

*function*

```python
def integrate(trace_map, *, connections=(), max_step, device, amplitude=0.0, flow_target=None, humidity_target=None, checkpoint=None, stop_index=None, species=None)
```

Integrate local growth and exchange on a recorded fixed mesh.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/_parcel_experiment.py#L338-L512)
