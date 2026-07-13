# Phase Details

Phase issue creation is intentionally deferred until E4 implementation issues
are generated and scheduled; `TBD` is not an unresolved design decision.

- [x] **E4-F2-P1:** Ideal and kappa activity Warp helpers with unit tests
  - Issue: #1287 | Size: S | Status: Implemented
  - Delivered: guarded fp64 ideal-molar and kappa water-activity device
    formulas as internal concrete-module helpers; no configuration, export,
    condensation-kernel, or transfer changes.
  - Files: `particula/gpu/dynamics/condensation_funcs.py` (new
      `water_activity_ideal_wp()` and `water_activity_kappa_wp()` helpers),
      `particula/gpu/dynamics/tests/condensation_funcs_test.py`
  - Tests: collection-safe Warp imports and independent NumPy parity for
    ideal pure/mixed/zero-total/water-free/nonzero-index cases and kappa
    wet/pure-water/dry/multi-solute/zero-kappa/nonzero-index cases.
- [x] **E4-F2-P2:** Static and composition-weighted surface physics with unit tests
  - Issue: #1288 | Size: S | Status: Shipped
  - Delivered: internal fp64 `effective_surface_tension_wp()` with exact static
    requested-species selection and a global single-phase composition-volume
    weighted scalar. Weighted mode ignores the requested index and uses the
    arithmetic mean of supplied tensions for zero total volume.
  - Files: `particula/gpu/dynamics/condensation_funcs.py`,
    `particula/gpu/dynamics/tests/condensation_funcs_test.py`
  - Tests: static selection/composition independence, independent NumPy parity
    for one-species/pure/mixed weighted cases, zero-volume mean fallback and
    weighted index independence, plus Kelvin radius/term consumption parity.
- [x] **E4-F2-P3:** Activity-adjusted Kelvin integration and validation tests
  - Issue: #1289 | Size: S | Status: Shipped
  - Delivered: `CondensationActivitySurfaceConfig` and keyword-only
    `activity_surface=` integrate ideal/kappa water-only activity and
    static/composition-weighted tension into GPU condensation. Aggregate
    validation is atomic and precedes all normalization, allocation, refresh,
    launch, and caller-state mutation; weighted mode uses one per-active-particle
    reduction buffer.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/environment.py`,
    `particula/gpu/kernels/tests/_condensation_test_support.py`,
    `particula/gpu/kernels/tests/condensation_test.py`
  - Tests: independent all-mode references, multi-box refresh/composition,
    legacy regression, frozen sidecar, edge cases, and monkeypatched no-launch
    atomic-failure snapshots.
- [x] **E4-F2-P4:** CPU and optional CUDA parity fixtures plus documentation
  - Issue: #1290 | Size: S | Status: Completed
  - Delivered: deterministic fp64 independent NumPy evidence for all four
    ideal/kappa × static/composition-weighted direct-step combinations over
    one-box and multi-box fixtures, including raw-transfer clamp semantics,
    vapor-pressure refresh, and unchanged gas ownership. Documentation now
    defines the shipped direct-kernel boundary and deferred CPU-only strategies.
  - Files: `particula/gpu/kernels/tests/_condensation_test_support.py`,
    `particula/gpu/kernels/tests/condensation_test.py`,
    `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`
  - Tests: Warp CPU is required when installed; separately marked CUDA coverage
    is availability-guarded. Fixtures exercise static and composition-weighted
    tension, ideal and kappa activity, one-/multi-box inputs, clamp-to-zero,
    refreshed vapor pressure, and unchanged gas concentration.
