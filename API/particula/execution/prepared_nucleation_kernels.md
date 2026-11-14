# particula.execution.prepared_nucleation_kernels

*module*

Device preparation for live fixed-capacity nucleation admission.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_nucleation_kernels.py#L1-L1)

## particula.execution.prepared_nucleation_kernels._releasable_slots

*function*

```python
def _releasable_slots(counts: wp.array2d(dtype=wp.float64), budget: wp.array(dtype=wp.int32))
```

Resampling can retain one active representative and release the rest.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/prepared_nucleation_kernels.py#L8-L20)
