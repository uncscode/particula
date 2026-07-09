# E3-F2 Phase Details

- [x] **E3-F2-P1:** Add mixed NPF/droplet fixture and acceptance-rate diagnostics with unit tests
  - Issue: #1241 | Size: S | Status: Shipped 2026-07-09
  - Depends on: E3-F1 stabilizing persisted `rng_states` usage enough that the
    fixture and seeded diagnostics do not encode a soon-to-change repeated-step
    contract.
  - Goal: Build reproducible mixed-scale coverage and expose measured sampler
    acceptance for test/debug analysis without changing public APIs.
  - Files: `particula/gpu/kernels/tests/coagulation_test.py` only.
  - Shipped details: added `_make_mixed_npf_droplet_particle_data()`,
    `_brownian_coagulation_attempt_diagnostic_kernel(...)`, and
    `_collect_test_local_attempt_diagnostics(...)` as test-local helpers.
  - Tests: `test_mixed_npf_droplet_fixture_returns_float64_particle_data()`,
    `test_mixed_npf_droplet_fixture_converts_on_supported_warp_devices(device)`,
    `test_mixed_scale_diagnostic_reports_attempted_and_accepted_counts(device)`,
    `test_mixed_scale_acceptance_fraction_is_finite_and_nonnegative(device)`,
    and `test_mixed_scale_sparse_box_returns_zero_accepted_collisions(device)`.

- [x] **E3-F2-P2:** Prototype bounded mixed-scale sampling hardening with conservation tests
  - Issue: #1242 | Size: S | Status: Shipped 2026-07-09
  - Depends on: E3-F2-P1 establishing the fixture and acceptance diagnostics so
    any hardening attempt is measured against a captured baseline instead of a
    guessed failure mode.
  - Goal: Ship a minimal bounded pair-selection hardening path that improves or
    bounds proposal behavior without changing Brownian physics.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`.
  - Shipped details: replaced retry-based raw-index proposals with bounded
    active-particle rank selection in production, mirrored the same selector in
    the test-local diagnostic kernel, and preserved collision-capacity,
    collision-pair buffer, and caller-owned RNG-state contracts.
  - Tests: selector-validity coverage for both diagnostic and production pair
    prefixes, zero/one-active sparse regressions, exactly-two-active fallback,
    accepted-count bounds, and mixed-scale total-mass conservation.

- [ ] **E3-F2-P3:** Compare statistical correctness against current Brownian behavior
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E3-F2-P1 and E3-F2-P2. Statistical comparison should run only
    after the fixture exists and the candidate hardening path, if any, is stable
    enough to compare against the current Brownian baseline.
  - Goal: Demonstrate that selected behavior preserves expected Brownian rates
    within stochastic tolerances or document the bounded limitation.
  - Files: `particula/gpu/kernels/tests/coagulation_test.py`, optional
    benchmark or metric helper in `particula/gpu/tests/benchmark_test.py`.
  - Tests: Aggregate stochastic collision-rate checks, conservation checks,
    deterministic seeded/reused-RNG behavior aligned with E3-F1.

- [ ] **E3-F2-P4:** Document selected design or accepted mixed-scale limitation
  - Issue: TBD | Size: XS | Status: Not Started
  - Depends on: E3-F2-P3 producing the evidence-backed outcome so the roadmap
    records measured acceptance bounds, preserved behavior, or an explicitly
    accepted limitation instead of an interim prototype state.
  - Goal: Update roadmap or feature documentation with reproduction commands,
    observed acceptance bounds, and the chosen implementation decision.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`, optional focused
    feature note under `docs/Features/`.
  - Tests: Documentation link/format validation plus any focused reproduction
    commands recorded in the document.
