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
- `particula/gpu/tests/benchmark_helpers_test.py` for fast coverage of the
  opt-in benchmark helpers, skip gates, projection helper behavior, and bounded
  benchmark metadata capture added with P3.
- `particula/gpu/tests/benchmark_test.py` for the optional P3 throughput entry
  point guarded by `--benchmark` plus Warp/CUDA skip-safe behavior.
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
- P3 extends `mass_precision_metrics_test.py` with cached reconstruction error,
  CPU-reference mass-transfer deltas, mixed-scale smallest-particle thresholds,
  zero-total-mass and zero-volume warning-clean paths, and explicit clamp
  accounting assertions.
- P3 also adds `benchmark_helpers_test.py` so the benchmark opt-in surface can
  be validated quickly without requiring CUDA.
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

Fast default validation now lives in
`mass_precision_cases_test.py`, `mass_precision_metrics_test.py`, and
`benchmark_helpers_test.py`. Any longer GPU sweeps should remain in
`benchmark_test.py` behind the documented opt-in reproduction commands.

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
pytest particula/gpu/tests/benchmark_helpers_test.py -q
pytest particula/gpu/tests/mass_precision_cases_test.py \
  particula/gpu/tests/mass_precision_metrics_test.py \
  particula/gpu/tests/benchmark_helpers_test.py -q
```

For the currently shipped scope, the bounded study surface now includes three
fast modules: `mass_precision_cases_test.py` provides the reusable baseline
fixtures, `mass_precision_metrics_test.py` exercises the shipped P2/P3
comparison layer, and `benchmark_helpers_test.py` validates the optional
benchmark surface without requiring a CUDA runtime.
