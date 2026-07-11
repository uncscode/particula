# Testing Strategy

## Fast Unit and Hook Tests

- Shipped coverage in `particula/tests/pytest_marker_policy_test.py` verifies:
  - hook registration of `warp`, `cuda`, `gpu_parity`, and `stochastic`
  - exact parity between `PYTEST_MARKER_LINES` and `pyproject.toml`
  - default collection leaves non-benchmark GPU-policy markers unmodified
  - mixed `benchmark` + GPU-policy items receive only benchmark skipping
  - `--benchmark` remains the only registered pytest option with unchanged help
    text
- `particula/tests/benchmark_option_test.py` remains as focused regression
  coverage for benchmark option registration and benchmark-only collection
  gating.
- All shipped P1 tests are fake/stub-driven and do not require Warp or CUDA.

## Warp CPU Validation

- Later phases should run focused GPU tests on Warp CPU when Warp is installed,
  for example:
  - `pytest particula/gpu/tests/cuda_availability_test.py -q`
  - `pytest particula/gpu/kernels/tests/coagulation_test.py -q`
  - `pytest particula/gpu/kernels/tests/condensation_test.py -q`
  - `pytest particula/gpu/kernels/tests/environment_test.py -q`
- If Warp is not installed, Warp-marked modules should skip by
  `pytest.importorskip('warp')` rather than fail at import time.

## CUDA-if-available Validation

- This validation path was not changed in P1 because no device helper or CUDA
  option surface shipped yet.
- CUDA remains optional/local/manual for release validation until dedicated CUDA
  CI is available.

## Tolerance Validation

- Deterministic parity tests should use explicit `rtol`/`atol` and document why
  those thresholds are appropriate.
- Conservation tests should retain tight numerical tolerances and fail on hidden
  mass/number drift.
- Stochastic coagulation tests should aggregate over seeds/steps, compare to
  expected rates using `3 sigma` or established tolerance bands, and avoid exact
  per-seed equality across CPU, Warp CPU, and CUDA.

## Regression Guardrails

- Ensure `pytest` without special GPU options still succeeds or skips cleanly in
  CPU-only environments.
- Ensure benchmark tests remain gated by `--benchmark` and are not accidentally
  pulled into default parity runs.
- Ensure marker-vocabulary drift between `particula/conftest.py` and
  `pyproject.toml` fails through regression coverage before unknown-marker
  warnings reach downstream GPU test phases.
