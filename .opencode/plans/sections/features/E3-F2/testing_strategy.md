# E3-F2 Testing Strategy

## Co-Located Unit and Regression Tests

- Add tests in `particula/gpu/kernels/tests/coagulation_test.py` alongside the
  coagulation code they exercise.
- Each phase that adds diagnostics or implementation behavior includes its own
  tests; no standalone testing phase is planned.

## Required Coverage

- Mixed NPF/droplet fixture creation with explicit `np.float64` values.
- Acceptance-rate metric sanity checks for the current sampler via test-local
  attempted/accepted diagnostics.
- Mass conservation after applying collision pairs on mixed-scale populations.
- Stochastic Brownian-rate comparison using expected mean and sigma tolerances.
- RNG behavior across repeated steps according to E3-F1 seed-once semantics.
- Sparse-bin, zero/one active-particle, and `max_collisions` boundary behavior.

## Shipped P1 Coverage

- `test_mixed_npf_droplet_fixture_returns_float64_particle_data()` verifies the
  canonical `(1, 4, 1)` mixed-scale fixture, `np.float64` dtypes, and
  nanometer-to-droplet order-of-magnitude span.
- `test_mixed_npf_droplet_fixture_converts_on_supported_warp_devices(device)`
  validates Warp CPU and CUDA-if-available conversion round trips.
- `test_mixed_scale_diagnostic_reports_attempted_and_accepted_counts(device)`
  checks integer attempted/accepted diagnostics and parity with production
  `n_collisions` results from the same seeded setup.
- `test_mixed_scale_acceptance_fraction_is_finite_and_nonnegative(device)`
  checks positive attempted counts plus finite, non-negative acceptance
  fractions for the mixed-scale case.
- `test_mixed_scale_sparse_box_returns_zero_accepted_collisions(device)` covers
  the fewer-than-two-active-particles edge case without warning-producing
  divide-by-zero paths.

## Shipped P2 Coverage

- `particula/gpu/kernels/tests/coagulation_test.py` now includes mixed-scale
  selector-validity coverage that asserts both diagnostic and production
  accepted-pair prefixes satisfy `0 <= i < j < n_particles` and only reference
  input slots that were active before the step.
- Sparse and degenerate regressions now cover zero-active and one-active
  particle setups with zero accepted collisions plus finite, non-negative
  diagnostics.
- Exactly-two-active fallback coverage confirms the bounded selector accepts the
  only valid non-adjacent pair when two active particles remain.
- Mixed-scale bounds assertions verify accepted counts remain within
  `collision_pairs.shape[1]`, `max_collisions`, and `n_particles // 2`.
- Mixed-scale conservation coverage now checks total pre/post mass with
  `numpy.testing.assert_allclose(..., rtol=1.0e-12)` after applying accepted
  collisions.

## Shipped P3 Coverage

- `test_mixed_scale_brownian_collision_totals_match_expected_mean_within_sigma_tolerance(device)`
  runs fixed seeds `101-200` against the shipped `E3-F2-P2` bounded selector
  path and asserts the observed total stays within an explicit 3-sigma
  tolerance around the Brownian expected mean.
- `test_mixed_scale_repeated_seeded_runs_conserve_total_mass_even_with_zero_acceptance_trials(device)`
  verifies repeated seeded runs conserve total mass across the same mixed-scale
  setup, including seeds that produce zero accepted collisions.
- `test_mixed_scale_caller_owned_rng_states_advance_without_hidden_reseed(device)`
  confirms reused mixed-scale `rng_states` advance across repeated calls rather
  than being silently reset.
- `test_mixed_scale_initialize_rng_true_replays_seeded_state_and_outcome(device)`
  confirms explicit `initialize_rng=True` reproduces the same seeded RNG state,
  collision counts/pairs, and resulting particle fields.

## Device Coverage

- Run on Warp CPU by default.
- Parametrize CUDA with existing availability helpers and skip cleanly when CUDA
  is unavailable.

## Suggested Commands

```bash
pytest particula/gpu/kernels/tests/coagulation_test.py -q -k mixed_scale
pytest particula/gpu/kernels/tests/coagulation_test.py -q -k "mixed_scale or initialize_rng or rng_states or conservation" -Werror
pytest particula/gpu/kernels/tests/coagulation_test.py -q
pytest particula/gpu/tests/mass_precision_cases_test.py -q
```

If a benchmark or characterization test is added behind an opt-in marker, record
the exact command in documentation rather than requiring it in normal CI.
