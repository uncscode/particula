# Testing Strategy

Every implementation phase ships co-located tests in the same change. Coverage
thresholds are never lowered; changed code must retain at least 80% coverage.

## Per-Phase Coverage

- **P1 (delivered, #1422):** `particula/particles/tests/exhaustion_test.py`
  covers defaults (`True`/`False`), exact-bool rejection, frozen/tuple-backed
  records, the policy truth table and resampling-first precedence, exact
  diagnostics and sentinels, schema/value validation, later-invalid-box
  no-mutation, zero-box/zero-capacity cases, activation-prefix independence,
  and the independent float64 weighted-inventory oracle at `rtol=1e-12`,
  `atol=1e-30`. It intentionally contains no P1 commit, resampling/scaling,
  discovery, particle/RNG, GPU, or distribution-moment tests.
- **P2 (delivered, #1423):** `particula/particles/tests/exhaustion_test.py`
  covers deterministic detached plans and repeated application; sparse, full,
  multi-box, and zero-release fixed-capacity remaps; stable equal-strata tie
  ordering; exact release clearing; independent weighted conservation checks;
  strict diagnostic-bound and malformed-P1 rejection; and stale, overlapping,
  and later-box malformed-plan atomicity. The focused oracle independently
  verifies the equal-stratum remap without production remap helpers.
- **P3 (delivered, #1424):**
  `particula/gpu/kernels/tests/exhaustion_test.py` covers the exported direct
  entry point, deterministic Warp CPU parity, caller-buffer ownership,
  shape/dtype/device/value/count/nonaliasing rejection, staged-plan diagnostic
  failures that skip commit, fixed-slot clearing, and optional CUDA clean
  skips. The independent NumPy oracle checks the stable sort and interval-sweep
  remap without CPU exhaustion helpers.
- **P4 (delivered, #1425):**
  `particula/particles/tests/exhaustion_test.py` and
  `particula/gpu/kernels/tests/exhaustion_test.py` cover selected-row scaling,
  independent NumPy-oracle Warp parity, identities, diagnostic-only
  all-unselected and zero-demand calls, valid empty boxes, exact protected and
  sibling-row preservation, and intensive/extensive invariants at
  `rtol=1e-12`, `atol=1e-30`. Compact invalid matrices assert stable exception
  types/messages and snapshot atomicity for particle fields and every sidecar,
  including later-invalid Warp rows. Warp CPU is the baseline and one guarded
  CUDA representative row skips cleanly when unavailable.
- **P4 export surface (delivered, #1425):** CPU tests assert the helper is not
  on `particula.particles`; `particula/gpu/tests/kernel_exports_test.py` asserts
  the Warp helper is absent from `particula.gpu.kernels`, its `__all__`, and its
  lazy-symbol map.
- **P5:** Integration tests cover enough-capacity bypass, resampling-only,
  scaling-only, resampling-sufficient, resampling-then-scaling, both-off
  exhausted error, and unsatisfiable demand without any write or truncation.
- **P6 (delivered, #1427):**
  `particula/particles/tests/exhaustion_test.py` and
  `particula/gpu/kernels/tests/exhaustion_test.py` add independent NumPy
  float64 particle/source ledgers for multi-box/multi-species sparse, full,
  all-inactive no-op, valid-highest-release, and over-capacity cases. Fixed
  admitted-source fixtures are applied only after successful P2/P4 boundaries,
  proving per-box/per-species number, mass, and signed-charge accounting at
  `rtol=1e-12`, `atol=1e-30`; P4 additionally includes volume-inclusive
  `scale * pre + source` assertions. Remap parity and moment/mixing diagnostics
  remain separate from conservation checks; Warp CPU is the baseline and CUDA
  rows skip cleanly when unavailable. The change is test-only.
- **P7:** Link, import-snippet, equation, shape-table, and supported/deferred
  boundary validation.

## Assertions and Tolerances

- Integer counts, policy codes, indices, sentinels, identities, shapes, dtypes,
  devices, and untouched state are exact.
- Number/species-mass/charge inventory uses independent float64 reductions with
  explicit `rtol`/`atol` recorded in tests; mixed scales receive species-level
  checks so large particles cannot hide small-particle loss.
- Later resampling phases define named distribution moments and thresholds;
  P1 makes no distribution-preservation guarantee because it does not resample.
- Invalid calls snapshot particles, volume, requests, diagnostics, work buffers,
  and any RNG state before asserting equality.

Focused tests use the repository `*_test.py` convention. Full fast pytest, Ruff,
mypy, and documentation checks run before completion.
