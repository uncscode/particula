# Change Log

## 2026-07-08

- Created first-pass feature plan for E3-F5 under parent epic E3.
- Added five issue-sized phases covering pytest marker policy, device helper
  standardization, stochastic tolerance documentation, GPU kernel test adoption,
  and CUDA-optional release validation docs.
- Incorporated workflow context from prior E3 feature drafting and codebase
  research findings for current Warp device fixtures, CUDA skip behavior, and
  stochastic coagulation tolerance patterns.

## 2026-07-08 — Completeness Review

- Expanded success criteria into measurable pass/fail checks, evidence metrics,
  and a definition of done for marker registration, helper behavior, policy
  adoption, and CUDA-optional validation guidance.

## 2026-07-10

- Updated the feature plan to reflect shipped `E3-F5-P1` work from issue
  `#1257`.
- Recorded that `particula/conftest.py` now centralizes the shared marker
  vocabulary in `PYTEST_MARKER_LINES` and preserves `--benchmark` as the only
  pytest option.
- Recorded that `pyproject.toml` mirrors the same marker strings and that new
  regression coverage lives in `particula/tests/pytest_marker_policy_test.py`
  alongside the existing benchmark option tests.

## 2026-07-10 — P2 implementation update

- Recorded shipped `E3-F5-P2` work from issue `#1258`.
- Noted that `particula/gpu/tests/cuda_availability.py` now exports shared
  `CUDA_SKIP_REASON` while preserving `cuda_available()` and `warp_devices()`
  behavior.
- Recorded benchmark skip-helper adoption in
  `particula/gpu/tests/benchmark_test.py` and shared-message regression
  assertions in `particula/gpu/tests/benchmark_helpers_test.py` and
  `particula/gpu/tests/cuda_availability_test.py`.

## 2026-07-10 — P3 documentation update

- Recorded shipped `E3-F5-P3` work from issue `#1259`.
- Noted that the implementation was documentation-only and limited to
  `.opencode/guides/testing_guide.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- Captured the shipped shared GPU testing policy: explicit deterministic
  `rtol`/`atol`, tight conservation checks, aggregate stochastic expectations,
  Warp CPU as the default parity backend, and optional/local/manual CUDA
  coverage.
