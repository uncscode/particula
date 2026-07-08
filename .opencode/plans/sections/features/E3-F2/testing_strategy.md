# E3-F2 Testing Strategy

## Co-Located Unit and Regression Tests

- Add tests in `particula/gpu/kernels/tests/coagulation_test.py` alongside the
  coagulation code they exercise.
- Each phase that adds diagnostics or implementation behavior includes its own
  tests; no standalone testing phase is planned.

## Required Coverage

- Mixed NPF/droplet fixture creation with explicit `np.float64` values.
- Acceptance-rate metric sanity checks for the current and selected behavior.
- Mass conservation after applying collision pairs on mixed-scale populations.
- Stochastic Brownian-rate comparison using expected mean and sigma tolerances.
- RNG behavior across repeated steps according to E3-F1 seed-once semantics.
- Sparse-bin, zero/one active-particle, and `max_collisions` boundary behavior.

## Device Coverage

- Run on Warp CPU by default.
- Parametrize CUDA with existing availability helpers and skip cleanly when CUDA
  is unavailable.

## Suggested Commands

```bash
pytest particula/gpu/kernels/tests/coagulation_test.py -q
pytest particula/gpu/tests/mass_precision_cases_test.py -q
```

If a benchmark or characterization test is added behind an opt-in marker, record
the exact command in documentation rather than requiring it in normal CI.
