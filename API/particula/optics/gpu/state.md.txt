# particula.optics.gpu.state

*module*

Read live native particle arrays into the prepared optical workspace.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L1-L1)

## particula.optics.gpu.state._CUBE_ROOT

*attribute* · *module attribute*

```python
_CUBE_ROOT = wp.constant(wp.float64(1.0 / 3.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L19-L19)

## particula.optics.gpu.state._GAS

*attribute* · *module attribute*

```python
_GAS = wp.constant(wp.float64(GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L21-L21)

## particula.optics.gpu.state._RADIUS_FACTOR

*attribute* · *module attribute*

```python
_RADIUS_FACTOR = wp.constant(wp.float64(3.0 / 4.0 * pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L18-L18)

## particula.optics.gpu.state._TWO_PI

*attribute* · *module attribute*

```python
_TWO_PI = wp.constant(wp.float64(2.0 * pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L20-L20)

## particula.optics.gpu.state.DeviceOpticalBatch

*class*

```python
class DeviceOpticalBatch
```

Reduce on the owner stream and retain a fixed number of optical rows.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L163-L263)

### particula.optics.gpu.state.DeviceOpticalBatch._cases

*attribute* · *instance attribute*

```python
_cases = ParticleOpticalCases(particles, config, device=device, stream=owner._stream)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L179-L184)

### particula.optics.gpu.state.DeviceOpticalBatch._humidity

*attribute* · *instance attribute*

```python
_humidity = wp.empty((batch_size, boxes), dtype=wp.float64, device=device) if water_lane is not None else None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L190-L194)

### particula.optics.gpu.state.DeviceOpticalBatch._owner

*attribute* · *instance attribute*

```python
_owner = owner
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L188-L188)

### particula.optics.gpu.state.DeviceOpticalBatch._rows

*attribute* · *instance attribute*

```python
_rows = wp.empty((batch_size, groups, 9), dtype=wp.float64, device=device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L196-L198)

### particula.optics.gpu.state.DeviceOpticalBatch._shape

*attribute* · *instance attribute*

```python
_shape = (boxes, wave_count, 9)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L195-L195)

### particula.optics.gpu.state.DeviceOpticalBatch._statuses

*attribute* · *instance attribute*

```python
_statuses = wp.empty((batch_size, groups), dtype=wp.int32, device=device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L199-L201)

### particula.optics.gpu.state.DeviceOpticalBatch._water_lane

*attribute* · *instance attribute*

```python
_water_lane = water_lane
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L189-L189)

### particula.optics.gpu.state.DeviceOpticalBatch._workspace

*attribute* · *instance attribute*

```python
_workspace = prepare_population_mie(self._cases.index, self._cases.size, self._cases.radius, self._cases.weight, particle_tile=particle_tile, coefficient_capacity=coefficient_capacity, stream=owner._stream)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L202-L210)

### particula.optics.gpu.state.DeviceOpticalBatch.__init__

*method*

```python
def __init__(owner, config, batch_size, particle_tile, coefficient_capacity, *, water_lane=None)
```

Bind device workspaces to the live particle arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L166-L210)

### particula.optics.gpu.state.DeviceOpticalBatch.download

*method*

```python
def download(count)
```

Check physics and optics before exposing any row of the batch.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L248-L263)

### particula.optics.gpu.state.DeviceOpticalBatch.enqueue

*method*

```python
def enqueue(row)
```

Append device results and status to a batch slot.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L212-L246)

## particula.optics.gpu.state.ParticleOpticalCases

*class*

```python
class ParticleOpticalCases
```

Prepare the optical inputs shared by resident observers and processes.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L107-L160)

### particula.optics.gpu.state.ParticleOpticalCases._fill

*attribute* · *instance attribute*

```python
_fill = wp.launch(_fill_cases, (groups, slots), inputs=[particles, wavelengths, indices, int(mixed), wp.float64(config.medium_index), self.index, self.size, self.radius, self.weight], device=device, stream=stream, record_cmd=True)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L139-L156)

### particula.optics.gpu.state.ParticleOpticalCases._inputs

*attribute* · *instance attribute*

```python
_inputs = (particles, wavelengths, indices)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L137-L137)

### particula.optics.gpu.state.ParticleOpticalCases._stream

*attribute* · *instance attribute*

```python
_stream = stream
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L138-L138)

### particula.optics.gpu.state.ParticleOpticalCases.index

*attribute* · *instance attribute*

```python
index = wp.empty((groups, slots), dtype=wp.vec2d, device=device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L132-L132)

### particula.optics.gpu.state.ParticleOpticalCases.__init__

*method*

```python
def __init__(particles, config, *, device, stream)
```

Allocate input buffers and retain a refresh command.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L110-L156)

### particula.optics.gpu.state.ParticleOpticalCases.enqueue

*method*

```python
def enqueue()
```

Fill optical coordinates from the live particle state.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L158-L160)

## particula.optics.gpu.state._fill_cases

*function*

```python
def _fill_cases(particles: WarpParticleData, wavelengths: wp.array(dtype=wp.float64), configured_index: wp.array2d(dtype=wp.vec2d), mixed: int, medium: wp.float64, index: wp.array2d(dtype=wp.vec2d), size: wp.array2d(dtype=wp.float64), radius: wp.array2d(dtype=wp.float64), weight: wp.array2d(dtype=wp.float64))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L24-L75)

## particula.optics.gpu.state._relative_humidity_row

*function*

```python
def _relative_humidity_row(vapor: wp.array2d(dtype=wp.float64), molar_mass: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), species: int, row: int, humidity: wp.array2d(dtype=wp.float64))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L92-L104)

## particula.optics.gpu.state._store_row

*function*

```python
def _store_row(result: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32), row: int, rows: wp.array3d(dtype=wp.float64), statuses: wp.array2d(dtype=wp.int32))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/state.py#L78-L89)
