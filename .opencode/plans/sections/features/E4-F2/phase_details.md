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
- [ ] **E4-F2-P3:** Activity-adjusted Kelvin integration and validation tests
  - Issue: TBD | Size: S | Status: Not Started
   - Goal: Pass numeric configuration through `condensation_step_gpu()` and
     compose activity, E4-F1 pure pressure, and Kelvin pressure in its transfer
     launch (keep this production-path delta to roughly 100 LOC).
   - Files: `particula/gpu/kernels/condensation.py`,
     `particula/gpu/kernels/tests/condensation_test.py`
  - Tests: Launch integration, invalid configuration, failure-before-mutation.
- [ ] **E4-F2-P4:** CPU and optional CUDA parity fixtures plus documentation
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Verify coupled physics and document supported/deferred behavior.
  - Files: GPU condensation tests, `docs/Features/`, plan sections
  - Tests: Warp CPU required, CUDA optional skip, docs/link validation.
