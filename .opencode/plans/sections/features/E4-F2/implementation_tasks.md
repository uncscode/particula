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
- [x] **E4-F2-P3 / issue #1289:** Added frozen keyword-only
  `CondensationActivitySurfaceConfig` and `activity_surface=` to
  `particula/gpu/kernels/condensation.py`, preserving the legacy positional
  per-species surface-tension API.
- [x] Validated the sidecar and all supplied aggregate inputs before environment
  normalization, allocations, vapor-pressure refresh, launch, or mutation.
- [x] Composed water-only ideal/kappa activity, E4-F1 pure pressure, and Kelvin
  tension in the transfer path; weighted tension is precomputed once per active
  particle in a step-owned fp64 buffer.
- [x] Retained fp64/fixed-shape storage and avoided host recomputation or
  transfers.

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
- [x] **E4-F2-P3 / issue #1289:** Added independent coupled references, all four
  mode-pair tests, multi-box and legacy regressions, frozen-sidecar coverage,
  edge cases, and no-launch/no-mutation validation snapshots in the co-located
  condensation kernel test support and test modules.
- [ ] **P4:** Record final CPU/CUDA parity evidence and user-facing
  supported/CPU-only physics documentation.
