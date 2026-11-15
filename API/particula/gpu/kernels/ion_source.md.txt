# particula.gpu.kernels.ion_source

*module*

Fixed-capacity paired ion source; the caller owns arrays and RNG words.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_source.py#L1-L1)

## particula.gpu.kernels.ion_source.inject_ion_pairs

*function*

```python
def inject_ion_pairs(masses: wp.array3d(dtype=wp.float64), counts: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64), rate: wp.float64, dt: wp.float64, positive_species: wp.int32, negative_species: wp.int32, positive_mass: wp.float64, negative_mass: wp.float64, rng: wp.array(dtype=wp.uint32), injected_pairs: wp.array(dtype=wp.int64), failure: wp.array(dtype=wp.int32))
```

Sample one pair count per box and reject insufficient capacity.

No fractional ions or resampling. The local RNG word is published only
after capacity admission. A fault in another box does not roll back boxes
that already completed; the owner must stop dependent processes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_source.py#L8-L65)
