# particula.optics.gpu

*module*

Explicit Warp optical kernels and prepared numerical workspaces.

Importing CPU optics does not load this namespace or initialize Warp.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/__init__.py#L1-L1)

## particula.optics.gpu.MieWorkspace

*class* · *dataclass*

```python
class MieWorkspace
```

Re-exported from: `particula.optics.gpu.workspace`

Prepared numerical outputs and retained device launch commands.

**Attributes**

- `efficiencies` (`wp.array`) — (cases,7): Qext,Qsca,Qabs,g,Qback,Qpr,scattering-valid.
- `status` (`wp.array`) — (cases,) coefficient status; zero means success.
- `perpendicular` (`wp.array | None`) — Optional (cases,angles) complex S1 as vec2d.
- `parallel` (`wp.array | None`) — Optional (cases,angles) complex S2 as vec2d.
- `mueller` (`wp.array | None`) — Optional (cases,angles,4,4) scaled Mueller matrices.
- `angular_status` (`wp.array | None`) — Optional per-angle amplitude status.
- `mueller_status` (`wp.array | None`) — Optional per-angle matrix status.
- `coefficient_capacity` (`int`) — Maximum orders retained in each tiled case.
- `case_tile` (`int`) — Maximum cases sharing reusable coefficient scratch.
- `angle_tile` (`int`) — Maximum angles in each angular launch.

Outputs are undefined until enqueue and successful check. Direct device
consumers may instead inspect status on-device. This object is not safe
for concurrent execution on multiple streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L37-L101)

### particula.optics.gpu.MieWorkspace._commands

*attribute* · *instance attribute*

```python
_commands: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L68-L68)

### particula.optics.gpu.MieWorkspace._device

*attribute* · *instance attribute*

```python
_device: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L70-L70)

### particula.optics.gpu.MieWorkspace._resources

*attribute* · *instance attribute*

```python
_resources: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L69-L69)

### particula.optics.gpu.MieWorkspace._stream

*attribute* · *instance attribute*

```python
_stream: wp.Stream | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L71-L71)

### particula.optics.gpu.MieWorkspace.angle_tile

*attribute* · *instance attribute*

```python
angle_tile: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L67-L67)

### particula.optics.gpu.MieWorkspace.angular_status

*attribute* · *instance attribute*

```python
angular_status: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L63-L63)

### particula.optics.gpu.MieWorkspace.case_tile

*attribute* · *instance attribute*

```python
case_tile: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L66-L66)

### particula.optics.gpu.MieWorkspace.coefficient_capacity

*attribute* · *instance attribute*

```python
coefficient_capacity: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L65-L65)

### particula.optics.gpu.MieWorkspace.efficiencies

*attribute* · *instance attribute*

```python
efficiencies: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L58-L58)

### particula.optics.gpu.MieWorkspace.mueller

*attribute* · *instance attribute*

```python
mueller: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L62-L62)

### particula.optics.gpu.MieWorkspace.mueller_status

*attribute* · *instance attribute*

```python
mueller_status: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L64-L64)

### particula.optics.gpu.MieWorkspace.parallel

*attribute* · *instance attribute*

```python
parallel: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L61-L61)

### particula.optics.gpu.MieWorkspace.perpendicular

*attribute* · *instance attribute*

```python
perpendicular: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L60-L60)

### particula.optics.gpu.MieWorkspace.status

*attribute* · *instance attribute*

```python
status: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L59-L59)

### particula.optics.gpu.MieWorkspace.__init__

*method*

```python
def __init__(efficiencies: wp.array, status: wp.array, perpendicular: wp.array | None, parallel: wp.array | None, mueller: wp.array | None, angular_status: wp.array | None, mueller_status: wp.array | None, coefficient_capacity: int, case_tile: int, angle_tile: int, _commands: tuple, _resources: tuple, _device: str, _stream: wp.Stream | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L0-L0)

### particula.optics.gpu.MieWorkspace.check

*method*

```python
def check() -> None
```

Synchronize and inspect status arrays; raise on numeric failure.

**Raises**

- `RuntimeError` — If any coefficient, angle or matrix case failed.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L82-L101)

### particula.optics.gpu.MieWorkspace.enqueue

*method*

```python
def enqueue() -> None
```

Enqueue retained commands, without allocation or host readback.

Callers order their input writes and consumers on the prepared stream.
Numeric failures remain in status buffers until explicit observation.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L73-L80)

## particula.optics.gpu.PopulationWorkspace

*class* · *dataclass*

```python
class PopulationWorkspace
```

Re-exported from: `particula.optics.gpu.bulk`

Retained streaming population outputs, commands and scratch.

**Attributes**

- `optics` (`wp.array`) — (groups,9): extinction, scattering, absorption, backscatter, radiation pressure [m⁻¹], SSA, g, extinction/scattering validity.
- `mueller` (`wp.array`) — (groups,angles,4,4) differential matrices [m⁻¹/sr]; zero length angle axis when angular output was not requested.
- `status` (`wp.array`) — Per-group status; 0 success, 2 input/upstream error, 3 overflow.
- `particle_tile` (`int`) — Fixed maximum live particle scratch count.

Groups flatten box/wavelength axes. Outputs require enqueue and successful
observation; inputs may be updated in place on the prepared stream.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L18-L56)

### particula.optics.gpu.PopulationWorkspace._commands

*attribute* · *instance attribute*

```python
_commands: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L38-L38)

### particula.optics.gpu.PopulationWorkspace._device

*attribute* · *instance attribute*

```python
_device: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L40-L40)

### particula.optics.gpu.PopulationWorkspace._resources

*attribute* · *instance attribute*

```python
_resources: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L39-L39)

### particula.optics.gpu.PopulationWorkspace._stream

*attribute* · *instance attribute*

```python
_stream: wp.Stream | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L41-L41)

### particula.optics.gpu.PopulationWorkspace.mueller

*attribute* · *instance attribute*

```python
mueller: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L35-L35)

### particula.optics.gpu.PopulationWorkspace.optics

*attribute* · *instance attribute*

```python
optics: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L34-L34)

### particula.optics.gpu.PopulationWorkspace.particle_tile

*attribute* · *instance attribute*

```python
particle_tile: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L37-L37)

### particula.optics.gpu.PopulationWorkspace.status

*attribute* · *instance attribute*

```python
status: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L36-L36)

### particula.optics.gpu.PopulationWorkspace.__init__

*method*

```python
def __init__(optics: wp.array, mueller: wp.array, status: wp.array, particle_tile: int, _commands: tuple, _resources: tuple, _device: str, _stream: wp.Stream | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L0-L0)

### particula.optics.gpu.PopulationWorkspace.check

*method*

```python
def check() -> None
```

Explicitly synchronize and reject failed population results.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L48-L56)

### particula.optics.gpu.PopulationWorkspace.enqueue

*method*

```python
def enqueue() -> None
```

Enqueue the retained streaming calculation without host transfers.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/bulk.py#L43-L46)

## particula.optics.gpu.prepare_mie

*function*

```python
def prepare_mie(relative_index: wp.array, size_parameter: wp.array, *, angles: wp.array | None = None, matrix_scale: wp.array | None = None, coefficient_capacity: int = 10090, case_tile: int = 256, angle_tile: int = 128, stream=None) -> MieWorkspace
```

Re-exported from: `particula.optics.gpu.workspace`

Prepare tiled device optics for already-uploaded numeric inputs.

**Parameters**

- `relative_index` (`wp.array`) — Contiguous (cases,) vec2d relative n+ik on CPU/CUDA.
- `size_parameter` (`wp.array`) — Contiguous (cases,) float64 x, same device.
- `angles` (`wp.array | None`) (default: `None`) — Optional contiguous (angles,) float64 radians, same device.
- `matrix_scale` (`wp.array | None`) (default: `None`) — Optional (cases,) float64 multiplier for Mueller values; default one. Only accepted when angles are requested.
- `coefficient_capacity` (`int`) (default: `10090`) — Positive scratch order bound, at most 10090. Insufficient capacity reports per-case status rather than truncates.
- `case_tile` (`int`) (default: `256`) — Positive maximum cases in coefficient scratch (default 256).
- `angle_tile` (`int`) (default: `128`) — Positive angular launch width (default 128).
- `stream` (default: `None`) — Optional CUDA stream on the same device. Input writes and downstream consumers must use this stream or explicit ordering.

**Returns**

- (`MieWorkspace`) — Workspace retaining inputs, output arrays, scratch and launch commands.
- (`MieWorkspace`) — Scratch scales with case_tile times coefficient_capacity; requested
- (`MieWorkspace`) — per-particle angular outputs still scale with cases times angles.
- (`MieWorkspace`) — No host values are copied, and preparation launches no scientific work.

**Raises**

- `ValueError` — If input metadata, device, stream or capacities conflict.
- `RuntimeError` — If the explicitly requested device cannot be used.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L154-L313)

## particula.optics.gpu.prepare_population_mie

*function*

```python
def prepare_population_mie(relative_index: wp.array, size_parameter: wp.array, radius: wp.array, number_density_weights: wp.array, *, angles: wp.array | None = None, particle_tile: int = 256, angle_tile: int = 128, coefficient_capacity: int = 10090, stream=None) -> PopulationWorkspace
```

Re-exported from: `particula.optics.gpu.bulk`

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

**Modules**

- `particula.optics.gpu._bulk`
- `particula.optics.gpu._complex`
- `particula.optics.gpu.angular`
- `particula.optics.gpu.bulk`
- `particula.optics.gpu.column`
- `particula.optics.gpu.cross_sections`
- `particula.optics.gpu.mie`
- `particula.optics.gpu.population`
- `particula.optics.gpu.workspace`
