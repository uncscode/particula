# Testing Strategy

Every implementation phase ships with co-located `*_test.py` coverage. Existing
coverage thresholds remain unchanged and changed code must retain at least 80%
coverage.

## Per-Phase Coverage

- **P1 (shipped, issue #1292):** Co-located executable coverage in
  `particula/gpu/kernels/tests/_condensation_test_support.py`, exported through
  `condensation_test.py`, verifies complete and partial `CondensationScratchBuffers`
  sidecars. It covers identity, stable shapes, raw work/total agreement,
  scalar/direct/hybrid/environment input forms, legacy positional compatibility,
  property-only sidecars with `mass_transfer`, and no required stable-shape
  allocations for a complete sidecar. Rejection coverage asserts wrong type,
  shape, fp64 dtype, device, and transfer overlap fail before launch,
  allocation/normalization/refresh instrumentation, or mutation.
- **P2 (shipped, issue #1293):** Co-located production tests verify four
  unconditional equal substeps, ordered E4-F1 refresh and environment
  preparation on every iteration, and transfer calculations from updated mass.
  They cover per-step clamping, accumulated applied totals versus final raw work
  proposals, forced evaporation, deterministic repeatability, finite
  nonnegative particle mass, unchanged gas concentration, and supported scalar,
  direct, hybrid, and explicit-environment inputs.
- **P3 (shipped, issue #1294):**
  `particula/gpu/kernels/tests/condensation_stiffness_test.py` exports only
  `test_condensation_production_stiffness_` support tests. The cached Warp CPU
  recorded grid covers nanometer, accumulation-mode, and two-box droplet-like
  cases against the independent fixed-four NumPy reference with the
  case-specific `rtol=5e-2` and maximum-relative-error bound. It separately
  checks unchanged gas concentration, stale-input vapor-pressure refresh,
  complete scratch identity/reuse, finite/nonnegative and zero-mass behavior,
  nonzero transfer, and exact repeatability. One `cuda`/`gpu_parity` droplet
  slice is additive and cleanly skipped when CUDA is unavailable.
- **P4:** Validate Markdown links, focused pytest commands, and consistency
  between roadmap claims and executable test names.

## Regression Boundaries

- Retain scalar, direct Warp-array, hybrid, and explicit environment coverage.
- Retain E4-F1 parameter, shape, species-order, device, positivity, and finite
  validation signals with failure before mutation.
- Distinguish the recorded stiffness evidence (`5e-2`) from tight conservation
  or deterministic equality tolerances.
- Do not treat gas conservation as E4-F3 acceptance; gas remains unchanged.

Focused command: `pytest particula/gpu/kernels/tests/condensation_test.py particula/gpu/kernels/tests/condensation_stiffness_test.py -q`.
