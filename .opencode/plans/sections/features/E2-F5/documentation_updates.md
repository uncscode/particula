# Documentation Updates

## Required Updates

- Update kernel docstrings to describe the P1 scalar compatibility bridge and
  reserved explicit `environment=` path.
- Document accepted P1 call forms for condensation and coagulation GPU APIs:
  scalar temperature/pressure is supported; mixed scalar-plus-environment and
  pure explicit-environment calls raise early `ValueError`.
- Note that temperature and pressure remain environment state, not `GasData`
  fields.

## Files Updated in P1

- Docstrings in `particula/gpu/kernels/condensation.py` and
  `particula/gpu/kernels/coagulation.py`.

## Deferred Documentation

- Broader roadmap or user-facing GPU docs remain deferred until later phases
  implement real per-box environment execution, normalization, and validation.

## Developer Notes

- Explain scalar compatibility as a migration aid for existing callers.
- Explain that P1 reserves, but does not yet execute, explicit per-box
  environment input.
