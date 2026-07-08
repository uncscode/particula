# Testing Strategy

## Fast Unit and Hook Tests

- Add tests for pytest marker/option registration and collection behavior using
  the same style as existing benchmark option tests.
- Add helper unit tests for device enumeration and CUDA skip behavior using fake
  Warp objects; do not require real CUDA for these tests.
- Verify missing CUDA produces deterministic skips rather than failures.

## Warp CPU Validation

- Run focused GPU tests on Warp CPU when Warp is installed, for example:
  - `pytest particula/gpu/tests/cuda_availability_test.py -q`
  - `pytest particula/gpu/kernels/tests/coagulation_test.py -q`
  - `pytest particula/gpu/kernels/tests/condensation_test.py -q`
  - `pytest particula/gpu/kernels/tests/environment_test.py -q`
- If Warp is not installed, Warp-marked modules should skip by
  `pytest.importorskip('warp')` rather than fail at import time.

## CUDA-if-available Validation

- On machines with CUDA available to Warp, the same device-parametrized tests
  should include `device='cuda'` automatically.
- On machines without CUDA, CUDA-specific branches should call the standardized
  skip helper and report clear skip reasons.
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
