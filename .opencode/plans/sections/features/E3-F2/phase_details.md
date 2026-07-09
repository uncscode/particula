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

- [x] **E3-F2-P3:** Compare statistical correctness against current Brownian behavior
  - Issue: #1243 | Size: S | Status: Shipped 2026-07-09
  - Depends on: E3-F2-P1 and E3-F2-P2. The shipped comparison reused the
    canonical mixed-scale fixture and bounded-selector implementation from the
    earlier phases instead of introducing a new kernel path.
  - Goal: Demonstrate that selected behavior preserves expected Brownian rates
    within stochastic tolerances and record the measured outcome with an exact
    reproduction command.
  - Files: `particula/gpu/kernels/tests/coagulation_test.py`,
    `docs/Features/Roadmap/data-oriented-gpu.md`.
  - Shipped details: added repeated seeded mixed-scale Brownian statistical
    checks for fixed seeds `101-200`, repeated-run total-mass conservation
    coverage that explicitly tolerates zero-acceptance trials, and mixed-scale
    caller-owned `rng_states` reuse/reset tests aligned with E3-F1 semantics.
  - Evidence recorded: the roadmap note captures the exact command
    `pytest particula/gpu/kernels/tests/coagulation_test.py -q -k mixed_scale`
    plus the measured Warp CPU result of 139 accepted collisions versus a
    Brownian expected mean of 143.846 with sigma 11.994.
  - Tests:
    `test_mixed_scale_brownian_collision_totals_match_expected_mean_within_sigma_tolerance(device)`,
    `test_mixed_scale_repeated_seeded_runs_conserve_total_mass_even_with_zero_acceptance_trials(device)`,
    `test_mixed_scale_caller_owned_rng_states_advance_without_hidden_reseed(device)`,
    and
    `test_mixed_scale_initialize_rng_true_replays_seeded_state_and_outcome(device)`.

- [x] **E3-F2-P4:** Document selected design or accepted mixed-scale limitation
  - Issue: #1244 | Size: XS | Status: Shipped 2026-07-09
  - Depends on: E3-F2-P3 producing the evidence-backed outcome so the roadmap
    records measured acceptance bounds, preserved behavior, or an explicitly
    accepted limitation instead of an interim prototype state.
  - Goal: Update roadmap or feature documentation with reproduction commands,
    observed acceptance bounds, and the chosen implementation decision.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md` only.
  - Shipped details: expanded the roadmap's mixed-scale known-issue note into a
    final decision record that says `E3-F2-P2` shipped bounded active-particle
    selector hardening inside the existing sampler, `E3-F2-P3` supplied the
    statistical/conservation evidence, and the remaining limitation is still the
    global-`k_max` plus one-thread-per-box acceptance boundary rather than an
    unresolved question about what design landed.
  - Documentation outcome: the note now names the private test-only fixture,
    diagnostic helpers, and key validation tests; preserves the exact measured
    139 versus 143.846 collision evidence with sigma 11.994 and 3-sigma
    tolerance 35.981; and keeps the reproduction commands explicit without
    implying any new production API or transfer-path behavior.
  - Tests: documentation readback plus the recorded focused commands
    `pytest particula/gpu/kernels/tests/coagulation_test.py -q -k mixed_scale`
    and `pytest particula/gpu/kernels/tests/coagulation_test.py -q -k
    "mixed_scale or sparse or degenerate or conservation" -Werror`.
