# particula.coupling.sounding_sequence

*module*

Time-dependent profile forcing for moving air and fixed finite volumes.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L1-L1)

## particula.coupling.sounding_sequence.ChangingSoundingFlow

*class* · *dataclass*

```python
class ChangingSoundingFlow(SpatialFlow)
```

Bases: `particula.coupling.flows.SpatialFlow`

Interpolate atmospheric profiles over level ground or a fixed dune.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L117-L271)

### particula.coupling.sounding_sequence.ChangingSoundingFlow.sequence

*attribute* · *instance attribute*

```python
sequence: SoundingSequence
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L121-L121)

### particula.coupling.sounding_sequence.ChangingSoundingFlow.terrain

*attribute* · *class attribute* · *instance attribute*

```python
terrain: tuple | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L122-L122)

### particula.coupling.sounding_sequence.ChangingSoundingFlow.__init__

*method*

```python
def __init__(sequence: SoundingSequence, terrain: tuple | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L0-L0)

### particula.coupling.sounding_sequence.ChangingSoundingFlow.__post_init__

*method*

```python
def __post_init__()
```

Check that the common profile spans the entire prescribed dune.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L124-L129)

### particula.coupling.sounding_sequence.ChangingSoundingFlow._provider

*method*

```python
def _provider(profile)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L131-L134)

### particula.coupling.sounding_sequence.ChangingSoundingFlow.interval

*method*

```python
def interval(grid, start, end, mixing=None, *, device='cpu')
```

Split at knots and face reversals before integrating linear flux.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L243-L271)

### particula.coupling.sounding_sequence.ChangingSoundingFlow.parcel_trace

*method*

```python
def parcel_trace(parcel_id, xyz, time, aerosol, *, provenance)
```

Integrate each linear time segment and retain the profile knots.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L148-L241)

### particula.coupling.sounding_sequence.ChangingSoundingFlow.reference_air

*method*

```python
def reference_air(grid, coord, time, device)
```

Sample pointwise atmosphere on the fixed grid geometry.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L142-L146)

### particula.coupling.sounding_sequence.ChangingSoundingFlow.sample

*method*

```python
def sample(xyz, time, *, device='cpu')
```

Sample world positions within the common profile coverage.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L136-L140)

### particula.coupling.sounding_sequence.ChangingSoundingFlow.trajectory

*method*

```python
def trajectory(parcel_id)
```

Inherited from: `particula.coupling.flows.FlowStrategy`

Return a recorded native parcel history by stable identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/flows.py#L36-L38)

## particula.coupling.sounding_sequence.SoundingSequence

*class* · *dataclass*

```python
class SoundingSequence
```

Profiles on a common height datum, timed in seconds since time_origin.

Profiles may have different vertical sampling. All queries use the
intersection of their height ranges. The caller supplies the common
height datum and measurement or synthetic origin in the provenance.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L13-L114)

### particula.coupling.sounding_sequence.SoundingSequence.profiles

*attribute* · *instance attribute*

```python
profiles: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L23-L23)

### particula.coupling.sounding_sequence.SoundingSequence.provenance

*attribute* · *instance attribute*

```python
provenance: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L25-L25)

### particula.coupling.sounding_sequence.SoundingSequence.time_origin

*attribute* · *instance attribute*

```python
time_origin: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L24-L24)

### particula.coupling.sounding_sequence.SoundingSequence.times

*attribute* · *instance attribute*

```python
times: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L22-L22)

### particula.coupling.sounding_sequence.SoundingSequence.__init__

*method*

```python
def __init__(times: tuple, profiles: tuple, time_origin: str, provenance: str) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L0-L0)

### particula.coupling.sounding_sequence.SoundingSequence.__post_init__

*method*

```python
def __post_init__()
```

Check that every profile shares a nonempty height/time domain.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L27-L72)

### particula.coupling.sounding_sequence.SoundingSequence.at

*method*

```python
def at(time)
```

Return the interpolated SI column at one time in seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L99-L114)

### particula.coupling.sounding_sequence.SoundingSequence.bracket

*method*

```python
def bracket(time)
```

Return the lower interval index, including at the last record.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L90-L97)

### particula.coupling.sounding_sequence.SoundingSequence.from_csv

*method* · *classmethod*

```python
def from_csv(path, *, time_origin, provenance)
```

Read profiles grouped by increasing time_s using named SI columns.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/sounding_sequence.py#L74-L88)
