# Implementation Tasks

## E3-F5-P1: Pytest markers and options

- [x] Inventory current markers in `particula/conftest.py` and
  `pyproject.toml`.
- [x] Add the concrete GPU/Warp marker definitions `warp`, `cuda`,
  `gpu_parity`, and `stochastic` in both locations so the proposed axes are now
  shared across hooks and static pytest config.
- [x] Decide whether a new option is needed beyond markers. P1 shipped with no
  new option; `--benchmark` remains the only registered pytest option and the
  only collection-affecting hook input.
- [x] Add hook regression tests for marker registration and option behavior in
  `particula/tests/pytest_marker_policy_test.py`, while keeping
  `particula/tests/benchmark_option_test.py` as the focused benchmark-policy
  regression file.

## E3-F5-P2: Device helper standardization

- Refactor `cuda_available()` and `warp_devices()` in
  `particula/gpu/tests/cuda_availability.py` only as needed to support the
  policy; keep backward-compatible names unless there is a strong reason not to.
- Add helper tests in `particula/gpu/tests/cuda_availability_test.py` with fake
  Warp objects for CPU-only and CUDA-available cases.
- Standardize CUDA skip messages and avoid real-CUDA requirements in unit
  tests; monkeypatch helper return values instead of probing hardware.

## E3-F5-P3: Tolerance documentation

- Add a device-aware Warp pytest policy subsection under the existing NVIDIA
  Warp tests section of `.opencode/guides/testing_guide.md`.
- Document stochastic parity examples from coagulation tests: aggregate over
  fixed seed ranges, use tolerance bands, and do not assert exact per-seed
  equality.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` to record the E3-F5 policy
  outcome, the required Warp CPU validation path, and CUDA's local/manual-only
  status.

## E3-F5-P4: Apply to GPU kernel tests

- Add markers to coagulation, condensation, environment, conversion, and other
  relevant GPU tests.
- Replace duplicated device fixtures with standardized helper usage where safe,
  starting with `particula/gpu/kernels/tests/coagulation_test.py`,
  `_condensation_test_support.py`, and `environment_test.py` before touching
  lower-value cleanup.
- Preserve module-level `pytest.importorskip("warp")` for Warp-dependent tests.
- Run focused Warp CPU tests and confirm CUDA paths either run or skip cleanly,
  recording any marker-selection command changes in the roadmap/testing docs.

## E3-F5-P5: Release validation docs

- Add focused commands for Warp CPU validation and optional CUDA validation in
  `.opencode/guides/testing_guide.md` and the GPU roadmap docs.
- Note CUDA remains local/manual before releases until CI gains CUDA capacity.
- Include troubleshooting notes for missing Warp, missing CUDA, and marker-based
  test selection so release validation steps stay deterministic for CPU-only
  contributors.

## Review Checklist

- New tests are co-located with each helper/hook change.
- No standalone testing-only phase is required; tests ship with implementation
  phases.
- Default CPU-only development remains unaffected.
