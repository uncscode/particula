# particula.optics.gpu.inverse

*module*

Bounded-memory CUDA bulk optical predictions and Warp tape adjoints.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L1-L1)

## particula.optics.gpu.inverse._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L17-L17)

## particula.optics.gpu.inverse.BulkOpticsAdjoint

*class*

```python
class BulkOpticsAdjoint
```

Bulk extinction, scattering and absorption with an analytic Warp VJP.

Construct for fixed (group, particle) dimensions and one vacuum
wavelength per group, allowing duplicates. Groups can represent
box/wavelength pairs or measurement windows. One workspace is used
on its device's current stream at a time.

Each evaluate() returns independent reduced arrays. With a Tape,
the reverse pass recomputes the local coefficients one tile at a time.
Inputs must stay alive and immutable through backward. Treat particle
count and coefficient series order as discrete. Only derivatives of the
smooth spherical optical model are supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L134-L319)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._count

*attribute* · *instance attribute*

```python
_count = groups * particles
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L179-L179)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._ext

*attribute* · *instance attribute*

```python
_ext = empty(tile, wp.vec3d)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L200-L200)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._index

*attribute* · *instance attribute*

```python
_index = empty(tile, wp.vec2d)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L190-L190)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._medium

*attribute* · *instance attribute*

```python
_medium = wp.float64(medium_index)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L184-L184)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._q

*attribute* · *instance attribute*

```python
_q = empty((tile, 7), wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L199-L199)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._sca

*attribute* · *instance attribute*

```python
_sca = empty(tile, wp.vec3d)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L201-L201)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._scale

*attribute* · *instance attribute*

```python
_scale = wp.array(2 * pi * medium_index / wave, dtype=wp.float64, device=self.device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L181-L183)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._series

*attribute* · *instance attribute*

```python
_series = (empty((tile, coefficient_capacity + 1), wp.vec2d), empty((tile, coefficient_capacity), wp.vec2d), empty((tile, coefficient_capacity), wp.vec2d), empty(tile, wp.int32), empty(tile, wp.int32))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L192-L198)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._size

*attribute* · *instance attribute*

```python
_size = empty(tile, wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L191-L191)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._tile

*attribute* · *instance attribute*

```python
_tile = min(self._count, particle_tile)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L180-L180)

### particula.optics.gpu.inverse.BulkOpticsAdjoint.device

*attribute* · *instance attribute*

```python
device = wp.get_device(device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L177-L177)

### particula.optics.gpu.inverse.BulkOpticsAdjoint.shape

*attribute* · *instance attribute*

```python
shape = shape
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L178-L178)

### particula.optics.gpu.inverse.BulkOpticsAdjoint.__init__

*method*

```python
def __init__(shape, wavelength, *, device='cuda:0', medium_index=1.0, particle_tile=256, coefficient_capacity=128)
```

Allocate fixed-capacity particle and Mie series scratch.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L149-L201)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._backward

*method*

```python
def _backward(radius, index, weight, bulk, failure, stream)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L289-L319)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._check

*method*

```python
def _check(bulk, failure)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L280-L287)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._evaluate_tile

*method*

```python
def _evaluate_tile(radius, index, weight, start)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L212-L234)

### particula.optics.gpu.inverse.BulkOpticsAdjoint._launch

*method*

```python
def _launch(kernel, *args)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L203-L210)

### particula.optics.gpu.inverse.BulkOpticsAdjoint.evaluate

*method*

```python
def evaluate(radius, index, weight, *, tape=None)
```

Return device bulk coefficients (group, 3) [m⁻¹].

Radius [m] and number-density weights [m⁻³] are float64. The
absolute refractive index is a vec2d of (n,k). All arrays are
contiguous with the prepared (group, particle) dimensions. To
record this operation, pass the same tape as the surrounding
Warp model. This call and the backward hook check device status.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L236-L278)

## particula.optics.gpu.inverse._gather

*function*

```python
def _gather(radius: wp.array2d(dtype=wp.float64), index: wp.array2d(dtype=wp.vec2d), weight: wp.array2d(dtype=wp.float64), scale: wp.array(dtype=wp.float64), medium: wp.float64, start: int, n_cases: int, gathered_index: wp.array(dtype=wp.vec2d), size: wp.array(dtype=wp.float64))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L20-L45)

## particula.optics.gpu.inverse._propagate

*function*

```python
def _propagate(radius: wp.array2d(dtype=wp.float64), weight: wp.array2d(dtype=wp.float64), q: wp.array2d(dtype=wp.float64), ext: wp.array(dtype=wp.vec3d), sca: wp.array(dtype=wp.vec3d), status: wp.array(dtype=wp.int32), scale: wp.array(dtype=wp.float64), medium: wp.float64, start: int, n_cases: int, adj_bulk: wp.array2d(dtype=wp.float64), adj_radius: wp.array2d(dtype=wp.float64), adj_index: wp.array2d(dtype=wp.vec2d), adj_weight: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L78-L131)

## particula.optics.gpu.inverse._reduce

*function*

```python
def _reduce(radius: wp.array2d(dtype=wp.float64), weight: wp.array2d(dtype=wp.float64), q: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32), start: int, n_cases: int, bulk: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/inverse.py#L48-L75)
