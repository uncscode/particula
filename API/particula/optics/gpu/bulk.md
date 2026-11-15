# particula.optics.gpu.bulk

*module*

Prepared bounded-memory optical reductions across particle populations.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L1-L1)

## particula.optics.gpu.bulk.PopulationWorkspace

*class* · *dataclass*

```python
class PopulationWorkspace
```

Retained streaming population outputs, commands and scratch.

**Attributes**

- `optics` (`wp.array`) — (groups,9): extinction, scattering, absorption, backscatter, radiation pressure [m⁻¹], SSA, g, extinction/scattering validity.
- `mueller` (`wp.array`) — (groups,angles,4,4) differential matrices [m⁻¹/sr]; zero length angle axis when angular output was not requested.
- `status` (`wp.array`) — Per-group status; 0 success, 2 input/upstream error, 3 overflow.
- `particle_tile` (`int`) — Fixed maximum live particle scratch count.

Groups flatten box/wavelength axes. Outputs require enqueue and successful
observation; inputs may be updated in place on the prepared stream.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L18-L56)

### particula.optics.gpu.bulk.PopulationWorkspace._commands

*attribute* · *instance attribute*

```python
_commands: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L38-L38)

### particula.optics.gpu.bulk.PopulationWorkspace._device

*attribute* · *instance attribute*

```python
_device: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L40-L40)

### particula.optics.gpu.bulk.PopulationWorkspace._resources

*attribute* · *instance attribute*

```python
_resources: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L39-L39)

### particula.optics.gpu.bulk.PopulationWorkspace._stream

*attribute* · *instance attribute*

```python
_stream: wp.Stream | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L41-L41)

### particula.optics.gpu.bulk.PopulationWorkspace.mueller

*attribute* · *instance attribute*

```python
mueller: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L35-L35)

### particula.optics.gpu.bulk.PopulationWorkspace.optics

*attribute* · *instance attribute*

```python
optics: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L34-L34)

### particula.optics.gpu.bulk.PopulationWorkspace.particle_tile

*attribute* · *instance attribute*

```python
particle_tile: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L37-L37)

### particula.optics.gpu.bulk.PopulationWorkspace.status

*attribute* · *instance attribute*

```python
status: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L36-L36)

### particula.optics.gpu.bulk.PopulationWorkspace.__init__

*method*

```python
def __init__(optics: wp.array, mueller: wp.array, status: wp.array, particle_tile: int, _commands: tuple, _resources: tuple, _device: str, _stream: wp.Stream | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L0-L0)

### particula.optics.gpu.bulk.PopulationWorkspace.check

*method*

```python
def check() -> None
```

Explicitly synchronize and reject failed population results.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L48-L56)

### particula.optics.gpu.bulk.PopulationWorkspace.enqueue

*method*

```python
def enqueue() -> None
```

Enqueue the retained streaming calculation without host transfers.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L43-L46)

## particula.optics.gpu.bulk.prepare_population_mie

*function*

```python
def prepare_population_mie(relative_index: wp.array, size_parameter: wp.array, radius: wp.array, number_density_weights: wp.array, *, angles: wp.array | None = None, particle_tile: int = 256, angle_tile: int = 128, coefficient_capacity: int = 10090, stream=None) -> PopulationWorkspace
```

Prepare streamed bulk optics without retaining particle optical output.

**Parameters**

- `relative_index` (`wp.array`) — Contiguous (groups,particles) vec2d relative n+ik.
- `size_parameter` (`wp.array`) — Same shape float64 x=2*pi*r*n_medium/lambda_vacuum.
- `radius` (`wp.array`) — Same shape float64 SI radius [m], consistent with x.
- `number_density_weights` (`wp.array`) — Same shape float64 density weights [m⁻³]. Callers supply discrete, resolved-count/volume or PDF weights.
- `angles` (`wp.array | None`) (default: `None`) — Optional shared 1D float64 angles [rad], on the same device.
- `particle_tile` (`int`) (default: `256`) — Maximum particles in scratch, default 256.
- `angle_tile` (`int`) (default: `128`) — Angular launch width, default 128.
- `coefficient_capacity` (`int`) (default: `10090`) — Maximum retained series order, default 10090.
- `stream` (default: `None`) — Optional matching CUDA stream; used for all work and checks.

**Returns**

- (`PopulationWorkspace`) — Population workspace. Scratch is bounded by particle_tile, series
- (`PopulationWorkspace`) — capacity and angle count, independently of total particle count.
- (`PopulationWorkspace`) — No CPU transfers, fallback or scientific launches during preparation.

**Raises**

- `ValueError` — If shapes/devices/capacities conflict. Physical-value errors
are reported by check after enqueue, without host input readback.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L59-L223)
