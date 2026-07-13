# Implementation Tasks

## Physics and Kernels
- [x] **E4-F2-P1 / issue #1287:** Added internal
  `water_activity_ideal_wp()` and `water_activity_kappa_wp()` in
  `particula/gpu/dynamics/condensation_funcs.py`. The fp64 helpers use the
  existing `(n_boxes, n_particles, n_species)` mass layout and explicit
  zero-total, zero-water, and zero-solute branches; no export or launch-path
  change was made.
- [x] **E4-F2-P2 / issue #1288:** Added internal fp64 Warp
  `effective_surface_tension_wp()` beside the activity helpers. Static mode
  selects the requested species without reading composition; weighted mode
  makes one species-axis pass over `mass / density`, ignores the requested
  index, and returns the arithmetic tension mean when total volume is zero.
- [ ] Extend `condensation_step_gpu()` with keyword-only numeric mode and
  parameter inputs while preserving legacy per-species surface input.
- [ ] Validate mode, water index, shape, dtype, device, order, positivity, and
  finiteness before mutation.
- [ ] In the `condensation_step_gpu()` transfer launch, compute particle pressure
  in the fixed order activity -> E4-F1 pure pressure -> Kelvin; keep this
  orchestration change bounded to roughly 100 production LOC.
- [ ] Retain fp64/fixed-shape storage and avoid host recomputation or transfer.

## Tooling / Tests
- [x] **E4-F2-P1 / issue #1287:** Repaired collection-safe Warp imports in
  `particula/gpu/dynamics/tests/condensation_funcs_test.py` and added
  independent NumPy parity tests. Coverage includes ideal pure, mixed,
  zero-total, water-free, and nonzero-water-index cases, plus kappa wet,
   pure-water, dry, multi-solute, zero-kappa, and nonzero-water-index cases.
- [x] **E4-F2-P2 / issue #1288:** Added co-located Warp kernels and independent
  NumPy fp64 references for static requested-species selection,
  composition-volume-weighted one-species/pure/mixed cases, zero-volume mean
  fallback, ignored weighted-mode indices, and Kelvin radius/term consumption.
- [ ] Extend independent NumPy reference fixtures for coupled pressure parity.
- [ ] Add one/multi-box kernel tests and static-input compatibility regression.
- [ ] Snapshot state around every expected validation failure.
- [ ] Run focused Warp CPU tests; parameterize CUDA with policy-compliant skips.
- [ ] Record `rtol`/`atol` and update supported/CPU-only physics documentation.
