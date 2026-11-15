# particula.gpu.kernels.ion_cleanup

*module*

Resident neutral ion-product export with deterministic per-box ledgers.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_cleanup.py#L1-L1)

## particula.gpu.kernels.ion_cleanup.is_neutral_ion_product

*function*

```python
def is_neutral_ion_product(masses: wp.array3d(dtype=wp.float64), counts: wp.array2d(dtype=wp.float64), charges: wp.array2d(dtype=wp.float64), positive_species: wp.int32, negative_species: wp.int32, box: wp.int32, slot: wp.int32) -> bool
```

Identify neutral ion-only products without selecting neutral aerosols.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_cleanup.py#L8-L31)

## particula.gpu.kernels.ion_cleanup.remove_neutral_ion_products

*function*

```python
def remove_neutral_ion_products(masses: wp.array3d(dtype=wp.float64), counts: wp.array2d(dtype=wp.float64), charges: wp.array2d(dtype=wp.float64), positive_species: wp.int32, negative_species: wp.int32, exported_mass: wp.array2d(dtype=wp.float64), exported_particles: wp.array(dtype=wp.int64), failure: wp.array(dtype=wp.int32), interval_steps: wp.int64, phase: wp.array(dtype=wp.int64))
```

Export only neutral products containing both reserved ion materials.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_cleanup.py#L34-L78)
