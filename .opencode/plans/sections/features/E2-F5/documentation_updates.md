# Documentation Updates

## Required Updates

- Update kernel docstrings to describe the shipped scalar/direct-array/
  explicit-environment normalization contract.
- Document accepted call forms for condensation and coagulation GPU APIs:
  scalar temperature/pressure, direct `(n_boxes,)` Warp arrays, hybrid
  scalar-plus-array direct inputs, and `WarpEnvironmentData`.
- Document stable early `ValueError` behavior for mixed direct-plus-environment,
  missing direct inputs, shape mismatch, and device mismatch failures.
- Note that temperature and pressure remain environment state, not `GasData`
  fields.

## Files Updated in This Phase

- `particula/gpu/kernels/environment.py` private helper docstrings.
- Docstrings in `particula/gpu/kernels/condensation.py` and
  `particula/gpu/kernels/coagulation.py`.

## Deferred Documentation

- Broader roadmap or user-facing GPU docs remain deferred until later phases
  need to expose more than the code-local entry-point contract.

## Developer Notes

- Explain scalar compatibility as a migration aid for existing callers.
- Explain that normalization is private to `particula.gpu.kernels` and does not
  expand the public conversion API in this phase.
