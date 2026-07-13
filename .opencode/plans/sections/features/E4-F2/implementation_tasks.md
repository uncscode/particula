# Implementation Tasks

## Physics and Kernels
- [ ] Add `water_activity_ideal_wp()` and `water_activity_kappa_wp()` to
  `particula/gpu/dynamics/condensation_funcs.py`, with CPU-equivalent dry
  guards, and cover each helper in `gpu/dynamics/tests/condensation_funcs_test.py`.
- [ ] Add `effective_surface_tension_wp()` beside those helpers; accept static
  per-species input and the selected composition-weighted mode, and define the
  zero-weight return explicitly in its focused test.
- [ ] Extend `condensation_step_gpu()` with keyword-only numeric mode and
  parameter inputs while preserving legacy per-species surface input.
- [ ] Validate mode, water index, shape, dtype, device, order, positivity, and
  finiteness before mutation.
- [ ] In the `condensation_step_gpu()` transfer launch, compute particle pressure
  in the fixed order activity -> E4-F1 pure pressure -> Kelvin; keep this
  orchestration change bounded to roughly 100 production LOC.
- [ ] Retain fp64/fixed-shape storage and avoid host recomputation or transfer.

## Tooling / Tests
- [ ] Add co-located formula tests for every supported mode and edge guard.
- [ ] Extend independent NumPy reference fixtures for coupled pressure parity.
- [ ] Add one/multi-box kernel tests and static-input compatibility regression.
- [ ] Snapshot state around every expected validation failure.
- [ ] Run focused Warp CPU tests; parameterize CUDA with policy-compliant skips.
- [ ] Record `rtol`/`atol` and update supported/CPU-only physics documentation.
