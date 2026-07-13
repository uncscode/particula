# Phase Details

Phase issue creation is intentionally deferred until E4 implementation issues
are generated and scheduled; `TBD` is not an unresolved design decision.

- [x] **E4-F1-P1:** Define thermodynamic model configuration and validation with unit tests
  - Issue: #1281 | Size: S | Status: Shipped
  - Delivered: validation-only fixed-shape sidecar and fail-early required
    condensation boundary; no formula, refresh, schema change, or formula launch.
  - Files: `particula/gpu/kernels/thermodynamics.py`, condensation boundary and
    support/tests, benchmark, and GPU direct-kernels quick-start.
  - Tests: valid/mutable mixed models; metadata/value/device/order failures;
    readback limits; and no-launch/no-mutation boundary regressions.

- [x] **E4-F1-P2:** Implement constant and Buck Warp vapor-pressure refresh with parity tests
  - Issue: #1282 | Size: S | Status: Shipped
  - Delivered: concrete-module-only `refresh_vapor_pressure_gpu` validates the
    Warp `float64` boundary, then performs one `(n_boxes, n_species)` launch to
    overwrite pressure using constant Pa values or canonical Buck water/ice
    equations; Buck parameter slots are reserved/unused.
  - Files: `particula/gpu/kernels/thermodynamics.py` and
    `particula/gpu/kernels/tests/thermodynamics_test.py`.
  - Tests: constant/Buck CPU parity below/at/above freezing, mixed models and
    multi-box overwrite, API export contract, and invalid-input no-mutation;
    Warp CPU parity with optional CUDA coverage.

- [ ] **E4-F1-P3:** Integrate pre-step refresh into GPU condensation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Refresh from normalized current temperature immediately before mass transfer.
  - Files: `particula/gpu/kernels/condensation.py`, condensation tests/support.
  - Tests: direct and environment inputs, temperature changes, positional compatibility, no host refresh.

- [ ] **E4-F1-P4:** Harden repeated-call and device contracts with integration tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prove reusable configuration/output behavior and failure-before-mutation guarantees.
  - Files: GPU thermodynamics/condensation modules and integration tests.
  - Tests: repeated calls, device mismatch, absent configuration, unchanged gas/particle buffers on error.

- [ ] **E4-F1-P5:** Document supported thermodynamic models and refresh ownership
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document modes, units, ownership, ordering, compatibility, and sibling boundaries.
  - Files: GPU feature/roadmap docs and plan status sections.
  - Tests: documentation link/reference validation.
