# Phase Details

Phase issue creation is intentionally deferred until E4 implementation issues
are generated and scheduled; `TBD` is not an unresolved design decision.

- [x] **E4-F4-P1:** Warp thermal-resistance helpers and validation with unit tests
  - Issue: #1297 | Size: S | Status: Shipped (2026-07-14)
  - Goal: Port the fp64 equations from `get_thermal_resistance_factor()` and
    `get_mass_transfer_rate_latent_heat()` before the E4-F3 calculate launch.
  - Delivered: private fp64 conductivity, thermal-factor, and corrected-rate
    helpers; atomic keyword-only `(n_species,)` `latent_heat` and
    `thermal_work` preflight. P1 validates but neither consumes nor mutates the
    sidecars, and does not change the production isothermal calculate launch.
  - Tests: `particula/gpu/dynamics/tests/condensation_funcs_test.py` covers CPU
    formula parity and exact zero-latent identity;
    `particula/gpu/kernels/tests/_condensation_test_support.py` covers valid
    sidecars, metadata/domain failures, and pre-mutation atomicity.

- [ ] **E4-F4-P2:** Per-substep latent-heat correction with parity tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Correct the common E4-F2 surface-pressure rate in all four substeps.
  - Files: `particula/gpu/kernels/condensation.py`
  - Tests: Corrected-rate parity, deterministic substeps, fallback, scratch reuse.

- [ ] **E4-F4-P3:** Signed whole-call energy bookkeeping with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Accumulate per-box/species energy from bounded applied transfer.
  - Files: `particula/gpu/kernels/condensation.py`
  - Tests: Sign, identity, clamp, aggregation, isolation, pre-mutation failure.

- [ ] **E4-F4-P4:** Warp integration regressions and documentation updates
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compose sibling contracts and document issue #1272 behavior.
  - Files: GPU condensation tests, `docs/Features/Roadmap/data-oriented-gpu.md`
  - Tests: Warp CPU integration, optional CUDA parity, docs validation.
