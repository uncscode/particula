# particula.coupling.flow_replay

*module*

Accepted grid intervals and native trajectory replay, with SI metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L1-L1)

## particula.coupling.flow_replay._UNITS

*attribute* · *module attribute*

```python
_UNITS = {'air': ['m/s', 'm/s', 'm/s', 'K', 'Pa', 'kg/kg dry air'], 'time': 's', 'coordinates': 'm', 'transfer': 'm3 outward', 'mixing': 'm3', 'diffusivity': 'm2/s along grid lines', 'speed': 'm/s reference'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L18-L26)

## particula.coupling.flow_replay.GridReplay

*class*

```python
class GridReplay(FlowStrategy)
```

Bases: `particula.coupling.flows.FlowStrategy`

Immutable forcing with fixed geometry and exact interval transfers.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L94-L213)

### particula.coupling.flow_replay.GridReplay.frames

*attribute* · *instance attribute*

```python
frames = tuple((validated_interval(grid, f) for f in frames))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L103-L103)

### particula.coupling.flow_replay.GridReplay.grid

*attribute* · *instance attribute*

```python
grid = grid
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L101-L101)

### particula.coupling.flow_replay.GridReplay.identity

*attribute* · *instance attribute*

```python
identity = digest.hexdigest()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L121-L121)

### particula.coupling.flow_replay.GridReplay.provenance

*attribute* · *instance attribute*

```python
provenance = provenance
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L102-L102)

### particula.coupling.flow_replay.GridReplay.times

*attribute* · *instance attribute*

```python
times = np.array([self.frames[0].start, *[f.end for f in self.frames]], dtype=np.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L109-L112)

### particula.coupling.flow_replay.GridReplay.__init__

*method*

```python
def __init__(grid, frames, provenance)
```

Validate field alignment, shared faces and accepted times.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L97-L121)

### particula.coupling.flow_replay.GridReplay.interval

*method*

```python
def interval(grid, start, end, mixing=None, *, device='cpu')
```

Return the exact next recorded interval on matching geometry.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L153-L167)

### particula.coupling.flow_replay.GridReplay.load

*method* · *classmethod*

```python
def load(path)
```

Read versioned numerical data without pickle.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L189-L213)

### particula.coupling.flow_replay.GridReplay.record

*method* · *classmethod*

```python
def record(grid, provider, times, mixing=None, *, provenance, device='cpu')
```

Record forcing through each endpoint, honoring short advances.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L123-L151)

### particula.coupling.flow_replay.GridReplay.save

*method*

```python
def save(path)
```

Write the forcing, geometry, conventions and provenance.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L169-L187)

### particula.coupling.flow_replay.GridReplay.trajectory

*method*

```python
def trajectory(parcel_id)
```

Inherited from: `particula.coupling.flows.FlowStrategy`

Return a recorded native parcel history by stable identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L36-L38)

## particula.coupling.flow_replay.LiveGrid

*class*

```python
class LiveGrid
```

A host-driven grid binding, retaining accepted forcing for replay.

Supply a matched recorded prefix when reopening a host restart. The source
ID identifies the host experiment and its authoritative field schema.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L216-L255)

### particula.coupling.flow_replay.LiveGrid.frames

*attribute* · *instance attribute*

```python
frames = list(history.frames) if history else []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L245-L245)

### particula.coupling.flow_replay.LiveGrid.grid

*attribute* · *instance attribute*

```python
grid = grid
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L231-L231)

### particula.coupling.flow_replay.LiveGrid.identity

*attribute* · *instance attribute*

```python
identity = digest.hexdigest()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L238-L238)

### particula.coupling.flow_replay.LiveGrid.provenance

*attribute* · *instance attribute*

```python
provenance = source_id
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L232-L232)

### particula.coupling.flow_replay.LiveGrid.start

*attribute* · *instance attribute*

```python
start = float(start)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L233-L233)

### particula.coupling.flow_replay.LiveGrid.times

*attribute* · *instance attribute*

```python
times = [self.start, *[f.end for f in self.frames]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L246-L246)

### particula.coupling.flow_replay.LiveGrid.__init__

*method*

```python
def __init__(grid, start, source_id, history=None)
```

Bind geometry and optionally restore its accepted history.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L223-L246)

### particula.coupling.flow_replay.LiveGrid.commit

*method*

```python
def commit(frame)
```

Retain the accepted immutable forcing after a successful step.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L252-L255)

### particula.coupling.flow_replay.LiveGrid.recording

*method*

```python
def recording()
```

Export the accepted history through the standard replay path.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L248-L250)

## particula.coupling.flow_replay.TrajectoryReplay

*class* · *dataclass*

```python
class TrajectoryReplay(FlowStrategy)
```

Bases: `particula.coupling.flows.FlowStrategy`

Offer one named trajectory using the existing CSV/JSON record.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L53-L63)

### particula.coupling.flow_replay.TrajectoryReplay.trace

*attribute* · *instance attribute*

```python
trace: ParcelTrace
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L57-L57)

### particula.coupling.flow_replay.TrajectoryReplay.__init__

*method*

```python
def __init__(trace: ParcelTrace) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L0-L0)

### particula.coupling.flow_replay.TrajectoryReplay.trajectory

*method*

```python
def trajectory(parcel_id)
```

Retrieve the recorded forcing without spatial extrapolation.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L59-L63)

### particula.coupling.flow_replay.TrajectoryReplay.interval

*method*

```python
def interval(grid, start, end, mixing, *, device='cpu')
```

Inherited from: `particula.coupling.flows.FlowStrategy`

Offer a grid interval or reject an incompatible provider.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L32-L34)

## particula.coupling.flow_replay.frozen_array

*function*

```python
def frozen_array(value, shape)
```

Validate, copy and freeze a finite float64 array.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L44-L50)

## particula.coupling.flow_replay.save_archive

*function*

```python
def save_archive(path, metadata, **arrays)
```

Atomically replace an archive after all arrays have been written.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L29-L41)

## particula.coupling.flow_replay.validated_interval

*function*

```python
def validated_interval(grid, f)
```

Own one accepted interval and check volume and mixing conservation.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flow_replay.py#L66-L91)
