# particula.gpu.kernels.cohort

*module*

Resident original-seed lineage bookkeeping, separate from particle counts.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/cohort.py#L1-L1)

## particula.gpu.kernels.cohort.initialize_cohort

*function*

```python
def initialize_cohort(masses: wp.array3d(dtype=wp.float64), counts: wp.array2d(dtype=wp.float64), seed_species: wp.int32, background_species: wp.int32, pending: wp.array2d(dtype=wp.int64), background: wp.array2d(dtype=wp.int32), statistics: wp.array2d(dtype=wp.int64))
```

Assign one original lineage per initial seed-containing particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/cohort.py#L8-L27)

## particula.gpu.kernels.cohort.merge_cohort

*function*

```python
def merge_cohort(pairs: wp.array3d(dtype=wp.int32), n_collisions: wp.array(dtype=wp.int32), pending: wp.array2d(dtype=wp.int64), background: wp.array2d(dtype=wp.int32), failure: wp.array(dtype=wp.int32))
```

Follow applied compacted pairs and clear donor labels for reuse.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/cohort.py#L30-L51)

## particula.gpu.kernels.cohort.observe_cohort

*function*

```python
def observe_cohort(masses: wp.array3d(dtype=wp.float64), counts: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), target_volume: wp.float64, pending: wp.array2d(dtype=wp.int64), background: wp.array2d(dtype=wp.int32), statistics: wp.array2d(dtype=wp.int64), failure: wp.array(dtype=wp.int32))
```

Count capture before size crossing, then enforce lineage partition.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/cohort.py#L54-L101)
