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

- [x] **E4-F1-P3:** Integrate pre-step refresh into GPU condensation with unit tests
  - Issue: #1283 | Size: S | Status: Shipped
  - Delivered: required keyword-only `ThermodynamicsConfig` now drives one
    pre-transfer refresh on every successful condensation call. The step
    validates all inputs first, casts direct `wp.float32` temperature into a
    device-local `wp.float64` buffer when required, refreshes caller-owned
    vapor pressure, then prepares environment properties and transfers mass.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/tests/_condensation_test_support.py`, and
    `particula/gpu/kernels/tests/condensation_test.py`.
  - Tests: launch ordering; scalar, direct, and explicit-environment inputs;
    repeated temperatures; float32 compatibility; signature compatibility; and
    pre-refresh failures with no refresh launch or gas/particle mutation.

- [x] **E4-F1-P4:** Harden repeated-call and device contracts with integration tests
  - Issue: #1284 | Size: S | Status: Shipped
  - Delivered: public-boundary integration coverage only; production code was
    unchanged because existing preflight ordering satisfied the atomicity
    contract.
  - Files: `particula/integration_tests/gpu_thermodynamics_contract_test.py`.
  - Tests: repeated reuse of one CPU-resident `ThermodynamicsConfig`,
    vapor-pressure matrix, and mass-transfer buffer; legacy positional
    `mass_transfer` compatibility; exact no-mutation snapshots for omitted
    configuration; and CUDA-only cross-device-sidecar atomicity.

- [x] **E4-F1-P5:** Document supported thermodynamic models and refresh ownership
  - Issue: #1285 | Size: XS | Status: Shipped
  - Delivered: documentation-only canonical contract for modes, units,
    ownership, ordering, compatibility, and sibling boundaries; no production
    code or supported-model expansion.
  - Files: GPU feature/roadmap docs and plan status sections.
  - Tests: documentation readback and focused thermodynamics/condensation
    regression validation.
