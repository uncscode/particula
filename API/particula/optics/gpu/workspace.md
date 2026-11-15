# particula.optics.gpu.workspace

*module*

Explicit preparation and observation for tiled resident optical kernels.

Preparation validates metadata, allocates storage and compiles retained launch
commands without advancing the calculation. Enqueue reuses those resources;
check is a deliberate synchronization/readback boundary. Input values may be
updated in place on the selected stream between calls; layouts stay fixed.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L1-L1)

## particula.optics.gpu.workspace.MieWorkspace

*class* · *dataclass*

```python
class MieWorkspace
```

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

### particula.optics.gpu.workspace.MieWorkspace._commands

*attribute* · *instance attribute*

```python
_commands: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L68-L68)

### particula.optics.gpu.workspace.MieWorkspace._device

*attribute* · *instance attribute*

```python
_device: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L70-L70)

### particula.optics.gpu.workspace.MieWorkspace._resources

*attribute* · *instance attribute*

```python
_resources: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L69-L69)

### particula.optics.gpu.workspace.MieWorkspace._stream

*attribute* · *instance attribute*

```python
_stream: wp.Stream | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L71-L71)

### particula.optics.gpu.workspace.MieWorkspace.angle_tile

*attribute* · *instance attribute*

```python
angle_tile: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L67-L67)

### particula.optics.gpu.workspace.MieWorkspace.angular_status

*attribute* · *instance attribute*

```python
angular_status: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L63-L63)

### particula.optics.gpu.workspace.MieWorkspace.case_tile

*attribute* · *instance attribute*

```python
case_tile: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L66-L66)

### particula.optics.gpu.workspace.MieWorkspace.coefficient_capacity

*attribute* · *instance attribute*

```python
coefficient_capacity: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L65-L65)

### particula.optics.gpu.workspace.MieWorkspace.efficiencies

*attribute* · *instance attribute*

```python
efficiencies: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L58-L58)

### particula.optics.gpu.workspace.MieWorkspace.mueller

*attribute* · *instance attribute*

```python
mueller: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L62-L62)

### particula.optics.gpu.workspace.MieWorkspace.mueller_status

*attribute* · *instance attribute*

```python
mueller_status: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L64-L64)

### particula.optics.gpu.workspace.MieWorkspace.parallel

*attribute* · *instance attribute*

```python
parallel: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L61-L61)

### particula.optics.gpu.workspace.MieWorkspace.perpendicular

*attribute* · *instance attribute*

```python
perpendicular: wp.array | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L60-L60)

### particula.optics.gpu.workspace.MieWorkspace.status

*attribute* · *instance attribute*

```python
status: wp.array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L59-L59)

### particula.optics.gpu.workspace.MieWorkspace.__init__

*method*

```python
def __init__(efficiencies: wp.array, status: wp.array, perpendicular: wp.array | None, parallel: wp.array | None, mueller: wp.array | None, angular_status: wp.array | None, mueller_status: wp.array | None, coefficient_capacity: int, case_tile: int, angle_tile: int, _commands: tuple, _resources: tuple, _device: str, _stream: wp.Stream | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L0-L0)

### particula.optics.gpu.workspace.MieWorkspace.check

*method*

```python
def check() -> None
```

Synchronize and inspect status arrays; raise on numeric failure.

**Raises**

- `RuntimeError` — If any coefficient, angle or matrix case failed.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L82-L101)

### particula.optics.gpu.workspace.MieWorkspace.enqueue

*method*

```python
def enqueue() -> None
```

Enqueue retained commands, without allocation or host readback.

Callers order their input writes and consumers on the prepared stream.
Numeric failures remain in status buffers until explicit observation.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L73-L80)

## particula.optics.gpu.workspace._admit

*function*

```python
def _admit(relative_index, size_parameter, angles, matrix_scale, coefficient_capacity, case_tile, angle_tile, stream)
```

Admit layouts and stream before allocating or compiling commands.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L117-L151)

## particula.optics.gpu.workspace._array

*function*

```python
def _array(value, dtype, shape, device, name)
```

Validate device array metadata without inspecting its device values.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L24-L34)

## particula.optics.gpu.workspace._capacities

*function*

```python
def _capacities(coefficient_capacity, case_tile, angle_tile)
```

Admit positive integer workspace capacities.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/workspace.py#L104-L114)

## particula.optics.gpu.workspace.prepare_mie

*function*

```python
def prepare_mie(relative_index: wp.array, size_parameter: wp.array, *, angles: wp.array | None = None, matrix_scale: wp.array | None = None, coefficient_capacity: int = 10090, case_tile: int = 256, angle_tile: int = 128, stream=None) -> MieWorkspace
```

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
