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

- [x] **E4-F4-P2:** Per-substep latent-heat correction with parity tests
  - Issue: #1298 | Size: S | Status: Shipped (2026-07-14)
  - Delivered: applies the correction in all four fixed substeps, sharing one
    activity/Kelvin surface pressure with pressure-delta logic. Omitted/all-zero
    latent heat preserves the isothermal path; `thermal_work` remains deferred.
  - Files: `particula/gpu/kernels/condensation.py`;
    `particula/gpu/kernels/tests/_condensation_test_support.py`.
  - Tests: CPU four-substep oracle/Warp parity, mixed and isolated-zero latent
    cases, rate reduction, validation atomicity, determinism, launch ordering,
    and scratch reuse/identity.

- [x] **E4-F4-P3:** Signed whole-call energy bookkeeping with unit tests
  - Issue: #1299 | Size: S | Status: Shipped (2026-07-14)
  - Delivered: optional caller-owned device-only `energy_transfer` output with
    atomic dependency/metadata validation, one post-preflight clear, and one
    post-four-substep single-writer box/species reduction of bounded whole-call
    mass transfer times latent heat. The two-item return and disabled path are
    unchanged.
  - Files: `particula/gpu/kernels/condensation.py`;
    `particula/gpu/kernels/tests/_condensation_test_support.py`;
    `particula/gpu/kernels/tests/condensation_test.py`.
  - Tests: overwrite/reuse, NaN/Inf output storage, oracle/clamp parity,
    box/species aggregation, atomic metadata failure, disabled behavior, and
    cleanly skippable CUDA parity.

- [ ] **E4-F4-P4:** Warp integration regressions and documentation updates
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compose sibling contracts and document issue #1272 behavior.
  - Files: GPU condensation tests, `docs/Features/Roadmap/data-oriented-gpu.md`
  - Tests: Warp CPU integration, optional CUDA parity, docs validation.
