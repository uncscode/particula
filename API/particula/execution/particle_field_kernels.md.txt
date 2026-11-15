# particula.execution.particle_field_kernels

*module*

Resident particle-field lifecycle kernels using native event decisions.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/particle_field_kernels.py#L1-L1)

## particula.execution.particle_field_kernels.birth_fields

*function*

```python
def birth_fields(values: wp.array3d(dtype=Any), initial: Any, counts: wp.array2d(dtype=wp.float64), selected: wp.array2d(dtype=wp.int32), selected_counts: wp.array(dtype=wp.int32), source: wp.array2d(dtype=wp.float64), additive: int, failure: wp.array(dtype=wp.int32))
```

Initialize actual activated slots, including reused resampling slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/particle_field_kernels.py#L56-L78)

## particula.execution.particle_field_kernels.merge_fields

*function*

```python
def merge_fields(values: wp.array3d(dtype=wp.float64), pairs: wp.array3d(dtype=wp.int32), collisions: wp.array(dtype=wp.int32), failure: wp.array(dtype=wp.int32))
```

Sum extensive per-particle quantities along applied compacted pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/particle_field_kernels.py#L8-L24)

## particula.execution.particle_field_kernels.preserve_activity

*function*

```python
def preserve_activity(before: wp.array2d(dtype=wp.float64), after: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Custom value updates must preserve particle slot activity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/particle_field_kernels.py#L144-L157)

## particula.execution.particle_field_kernels.remap_fields

*function*

```python
def remap_fields(values: wp.array3d(dtype=wp.float64), original: wp.array3d(dtype=wp.float64), entries: wp.array(dtype=wp.int32), targets: wp.array2d(dtype=wp.int32), donors: wp.array2d(dtype=wp.int32), overlaps: wp.array2d(dtype=wp.float64), retained: wp.array2d(dtype=wp.int32), replacement_counts: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Apply the exact native mass remap's recorded interval overlaps.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/particle_field_kernels.py#L81-L108)

## particula.execution.particle_field_kernels.retire_fields

*function*

```python
def retire_fields(values: wp.array3d(dtype=Any), zero: Any, before: wp.array2d(dtype=wp.float64), after: wp.array2d(dtype=wp.float64), removed: wp.array2d(dtype=wp.float64), additive: int, failure: wp.array(dtype=wp.int32))
```

Preserve per-particle values under count loss and clear retired slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/particle_field_kernels.py#L27-L53)

## particula.execution.particle_field_kernels.validate_fields

*function*

```python
def validate_fields(values: wp.array3d(dtype=Any), lower: Any, upper: Any, zero: Any, counts: wp.array2d(dtype=wp.float64), source: wp.array2d(dtype=wp.float64), removed: wp.array2d(dtype=wp.float64), additive: int, failure: wp.array(dtype=wp.int32))
```

Check active bounds, inactive clearing and finite inventory ledgers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/particle_field_kernels.py#L111-L141)
