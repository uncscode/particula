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
- Shipped P2 coverage in `particula/gpu/tests/cuda_availability_test.py`
  verifies targeted `_pack_` warning suppression, stable `warp_devices()`
  enumeration for CPU-only vs CUDA-available fake Warp objects, and exact
  `CUDA_SKIP_REASON` export.
- `particula/gpu/tests/benchmark_helpers_test.py` now verifies that
  `_skip_if_no_cuda()` reuses `CUDA_SKIP_REASON` when Warp is missing or CUDA is
  unavailable and returns normally when CUDA is reported available.

## Warp CPU Validation

- The shipped documentation policy now states that focused GPU parity tests run
  on Warp CPU by default when Warp is installed, for example:
  - `pytest particula/gpu/tests/cuda_availability_test.py -q`
  - `pytest particula/gpu/kernels/tests/coagulation_test.py -q`
  - `pytest particula/gpu/kernels/tests/condensation_test.py -q`
  - `pytest particula/gpu/kernels/tests/environment_test.py -q`
- If Warp is not installed, Warp-marked modules should skip by
  `pytest.importorskip('warp')` rather than fail at import time.

## CUDA-if-available Validation

- P2 standardized the shared CUDA-only skip message and P3 documented the same
  optional/local/manual CUDA policy in both canonical docs.
- CUDA remains optional/local/manual for release validation until dedicated CUDA
  CI is available, and standard CI must skip cleanly when CUDA is unavailable.

## Tolerance Validation

- Deterministic parity tests should use explicit `rtol`/`atol` and document why
  those thresholds are appropriate.
- Conservation tests should retain tight numerical tolerances and fail on hidden
  mass/number drift.
- Stochastic coagulation tests should aggregate over seeds/steps, compare to
  expected rates using `3 sigma` or established tolerance bands, and avoid exact
  per-seed equality across CPU, Warp CPU, and CUDA.

## Documentation Validation

- `E3-F5-P3` was documentation-only, so validation is a manual doc consistency
  check rather than a new Python test module.
- Re-opened `.opencode/guides/testing_guide.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md` should agree on marker names,
  Warp CPU default coverage, CUDA optionality, explicit deterministic
  `rtol`/`atol`, tight conservation checks, and aggregate stochastic
  expectations.

## Regression Guardrails

- Ensure `pytest` without special GPU options still succeeds or skips cleanly in
  CPU-only environments.
- Ensure benchmark tests remain gated by `--benchmark` and are not accidentally
  pulled into default parity runs.
- Ensure shared CUDA skip messaging stays centralized in
  `particula/gpu/tests/cuda_availability.py` so downstream GPU tests do not
  drift back to duplicated literals.
- Ensure marker-vocabulary drift between `particula/conftest.py` and
  `pyproject.toml` fails through regression coverage before unknown-marker
  warnings reach downstream GPU test phases.
