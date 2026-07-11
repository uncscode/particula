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

- [x] Keep `cuda_available()` and `warp_devices()` backward-compatible in
  `particula/gpu/tests/cuda_availability.py` while adding the shared
  `CUDA_SKIP_REASON` export.
- [x] Expand `particula/gpu/tests/cuda_availability_test.py` with fake-Warp
  coverage for warning suppression, CPU-only enumeration, CUDA enumeration, and
  the exact shared constant value.
- [x] Standardize the benchmark CUDA skip path to reuse `CUDA_SKIP_REASON` in
  `particula/gpu/tests/benchmark_test.py` and update
  `particula/gpu/tests/benchmark_helpers_test.py` to assert the shared message
  for missing-Warp, CUDA-unavailable, and non-skipping branches without probing
  real hardware.

## E3-F5-P3: Tolerance documentation

- [x] Add a device-aware Warp pytest policy subsection under the existing NVIDIA
  Warp tests section of `.opencode/guides/testing_guide.md`.
- [x] Document stochastic parity examples from coagulation tests: aggregate over
  fixed seed ranges, use tolerance bands, and do not assert exact per-seed
  equality.
- [x] Update `docs/Features/Roadmap/data-oriented-gpu.md` to record the E3-F5 policy
  outcome, the required Warp CPU validation path, and CUDA's local/manual-only
  status.

## E3-F5-P4: Apply to GPU kernel tests

- [x] Add markers to coagulation, condensation, environment, conversion, and other
  relevant GPU tests.
- [x] Replace duplicated device fixtures with standardized helper usage where safe,
  starting with `particula/gpu/kernels/tests/coagulation_test.py`,
  `_condensation_test_support.py`, and `environment_test.py` before touching
  lower-value cleanup.
- [x] Preserve module-level `pytest.importorskip("warp")` for Warp-dependent tests.
- [x] Run focused Warp CPU tests and confirm CUDA paths either run or skip cleanly,
  recording any marker-selection command changes in the roadmap/testing docs.

## E3-F5-P5: Release validation docs

- [x] Add focused commands for Warp CPU validation and optional CUDA validation in
  `.opencode/guides/testing_guide.md` and the GPU roadmap docs.
- [x] Note CUDA remains local/manual before releases until CI gains CUDA capacity.
- [x] Include troubleshooting notes for missing Warp, missing CUDA, and marker-based
  test selection so release validation steps stay deterministic for CPU-only
  contributors.
- [x] Leave `docs/contribute/CONTRIBUTING.md` unchanged because the testing guide
  and roadmap remain the canonical GPU policy homes for this docs-only phase.

## Review Checklist

- New tests are co-located with each helper/hook change.
- No standalone testing-only phase is required; tests ship with implementation
  phases.
- Default CPU-only development remains unaffected.
