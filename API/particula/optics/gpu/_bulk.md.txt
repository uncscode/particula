# particula.optics.gpu._bulk

*module*

Chunk gathering and accumulation for bounded-memory population optics.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_bulk.py#L1-L1)

## particula.optics.gpu._bulk._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(math.pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_bulk.py#L9-L9)

## particula.optics.gpu._bulk.accumulate_angular

*function*

```python
def accumulate_angular(matrices: wp.array4d(dtype=wp.float64), weights: wp.array(dtype=wp.float64), input_status: wp.array2d(dtype=wp.int32), group: int, totals: wp.array3d(dtype=wp.float64), status: wp.array(dtype=wp.int32))
```

Accumulate scaled chunk matrices without full-population storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_bulk.py#L104-L125)

## particula.optics.gpu._bulk.accumulate_scalar

*function*

```python
def accumulate_scalar(efficiencies: wp.array2d(dtype=wp.float64), radius: wp.array(dtype=wp.float64), weights: wp.array(dtype=wp.float64), gather_status: wp.array(dtype=wp.int32), optical_status: wp.array(dtype=wp.int32), group: int, totals: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32))
```

Accumulate cross sections and scattering-weighted g in fixed order.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_bulk.py#L78-L101)

## particula.optics.gpu._bulk.clear_bulk

*function*

```python
def clear_bulk(totals: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32), matrices: wp.array3d(dtype=wp.float64))
```

Reset accumulated state once at the beginning of each evaluation.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_bulk.py#L62-L75)

## particula.optics.gpu._bulk.finalize_bulk

*function*

```python
def finalize_bulk(totals: wp.array2d(dtype=wp.float64), status: wp.array(dtype=wp.int32))
```

Convert accumulated moments to SSA/g and validate finite results.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_bulk.py#L128-L147)

## particula.optics.gpu._bulk.gather_cases

*function*

```python
def gather_cases(indices: wp.array2d(dtype=wp.vec2d), sizes: wp.array2d(dtype=wp.float64), radii: wp.array2d(dtype=wp.float64), weights: wp.array2d(dtype=wp.float64), group: int, start: int, valid_count: int, index: wp.array(dtype=wp.vec2d), size: wp.array(dtype=wp.float64), radius: wp.array(dtype=wp.float64), weight: wp.array(dtype=wp.float64), scale: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32))
```

Fill a fixed-size chunk; padding is an exact index-matched zero case.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_bulk.py#L12-L59)
