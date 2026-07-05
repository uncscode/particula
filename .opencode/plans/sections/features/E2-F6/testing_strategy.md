# E2-F6 Testing Strategy

## Co-located Testing Policy

Each phase that adds helper functions or scripts must ship tests with that
phase. There is no standalone testing phase.

## Target Test Files

- `particula/gpu/tests/mass_precision_cases_test.py` for the shipped P1
  deterministic case generation, shape checks, finite-value assertions,
  malformed-input rejection, `ParticleData` compatibility, derived-radius
  checks, and CPU/Warp `fp64` baseline assertions.
- `particula/gpu/tests/mass_precision_metrics_test.py` for shipped P2
  candidate projection/reconstruction helpers, mass/radius fidelity checks,
  invalid-candidate coverage, zero-total-mass handling, unsupported-candidate
  doc-only coverage, and CPU/Warp dtype regression checks.
- `particula/gpu/tests/mass_precision_case_helpers.py` only if shared helper
  logic becomes large enough to justify a small adjacent test helper module.
- `docs/Features/Roadmap/mass-precision-study.md` for the reproducibility
  commands and evidence links finalized in P4.

## Unit and Reproducibility Tests

- `mass_precision_cases_test.py` now asserts deterministic output, expected
  shapes, finite values, nonnegative masses, malformed-input rejection, and
  physically reasonable radii for the shipped `npf_cluster`,
  `five_to_ten_nm`, `accumulation_mode`, and `cloud_droplet` cases.
- P2 candidate tests now cover conversion/reconstruction behavior for
  `fp32_absolute_mass`, `mixed_precision_mass_plus_density`, and
  `fp32_total_mass_fp32_mass_fraction`, with explicit mass and radius
  tolerance assertions against the `fp64` baseline.
- Production default tests should continue to assert `np.float64` and
  `wp.float64` in `particula/particles/tests/` and `particula/gpu/tests/`
  anywhere current behavior is intentionally unchanged; P1 added explicit
  baseline-policy and Warp round-trip coverage in
  `mass_precision_cases_test.py`, and P2 added focused dtype-regression
  coverage in `mass_precision_metrics_test.py`.

## Numerical Validation

- Compare all candidates against current absolute `fp64` reference values.
- Track absolute and relative mass error, radius error, and small-particle error
  when large droplets coexist in the same box.
- Use CPU conservation-limited mass-transfer functions for conservation
  validation when gas coupling is included.
- Record clamping frequency and mass introduced/removed by clamping for GPU
  condensation comparisons.

For P3, keep numerical assertions executable in
`particula/gpu/tests/mass_precision_metrics_test.py` rather than only in the
report so reviewers can rerun the same thresholds from the PR.

## Benchmark and GPU Tests

- Reuse existing benchmark markers: `slow`, `performance`, and `benchmark`.
- Use skip-safe CUDA/Warp device selection patterns from `particula/gpu/tests/`.
- Keep fast unit tests separate from slow evidence-generation benchmarks.

Fast default validation should stay in the two focused test files above. Any
longer GPU sweeps should either be marked slow in the same files or invoked from
documented reproduction commands in the study report.

## Documentation Validation

- The final report must include commands or scripts used to reproduce evidence.
- Documentation links should be validated as part of the final phase.

P4 is a docs/report phase, so its valid exception is documentation validation
plus rerunning the focused mass-precision tests that back the published tables;
it should not defer unresolved helper or metric tests into a later PR.

## Verification Commands

```bash
pytest particula/gpu/tests/mass_precision_cases_test.py -q
pytest particula/gpu/tests/mass_precision_metrics_test.py -q
pytest particula/gpu/tests/mass_precision_cases_test.py \
  particula/gpu/tests/mass_precision_metrics_test.py -q
```

For the currently shipped scope, both focused modules are part of the bounded
study surface: `mass_precision_cases_test.py` provides the reusable baseline
fixtures and `mass_precision_metrics_test.py` exercises the shipped P2
comparison layer.
