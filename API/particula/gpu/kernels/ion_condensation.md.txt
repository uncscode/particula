# particula.gpu.kernels.ion_condensation

*module*

Gate ion-only vapor proposals before condensation inventory reduction.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_condensation.py#L1-L1)

## particula.gpu.kernels.ion_condensation.exclude_ion_candidates

*function*

```python
def exclude_ion_candidates(candidate: wp.array3d(dtype=wp.float64), masses: wp.array3d(dtype=wp.float64), positive_species: wp.int32, negative_species: wp.int32)
```

Zero uptake and evaporation proposals for ion-only slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_condensation.py#L8-L25)

## particula.gpu.kernels.ion_condensation.gate_ion_condensation

*function*

```python
def gate_ion_condensation(prepared)
```

Apply the optional prepared filter without touching native state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/ion_condensation.py#L28-L41)
