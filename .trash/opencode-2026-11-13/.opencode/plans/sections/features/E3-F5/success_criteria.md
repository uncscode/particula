# Success Criteria

## Pass / Fail Criteria

- [x] The marker vocabulary is explicit and consistent across code, config, and
  docs: `warp`, `cuda`, `gpu_parity`, and `stochastic` are registered without
  unknown-marker warnings.
- [x] Warp CPU remains the required baseline for GPU-test development, while
  CUDA runs only when available and otherwise skips cleanly with reusable
  helper behavior.
- [x] Device-selection helpers and skip utilities are covered by focused tests
  that use fakes or monkeypatching rather than requiring real CUDA hardware.
- [x] Representative GPU kernel tests adopt the standardized markers/helpers
  without weakening existing coverage for condensation, coagulation,
  environment, or conversion paths.
- [x] The stochastic tolerance policy documents deterministic equality,
  conservation, and aggregate-probabilistic checks with concrete guidance such
  as `3 sigma`-style bands where appropriate.
- [x] Manual/release validation commands document CUDA-optional workflows and do
  not turn CUDA into a required CI gate.
- [x] Existing benchmark gating through `--benchmark` remains intact and is not
  conflated with default validation.

## Current Shipped Evidence (E3-F5-P1 to P5)

- `particula/conftest.py` and `pyproject.toml` now declare the same seven-marker
  vocabulary, including `warp`, `cuda`, `gpu_parity`, and `stochastic`.
- `particula/tests/pytest_marker_policy_test.py` proves marker parity, default
  non-benchmark collection behavior, mixed benchmark-plus-Warp handling, and the
  benchmark-only option/help-text surface.
- `particula/tests/benchmark_option_test.py` still covers the original
  `--benchmark` registration and benchmark gating path.
- `.opencode/guides/testing_guide.md` now records the shared device-aware GPU
  testing policy: Warp CPU default parity, optional/local/manual CUDA coverage,
  explicit deterministic `rtol`/`atol`, tight conservation checks, and
  aggregate stochastic expectations.
- `docs/Features/Roadmap/data-oriented-gpu.md` now mirrors the same shipped
  tolerance semantics and ties them to the mixed-scale stochastic evidence.
- Those two canonical docs now also carry the final release-validation wording:
  Warp CPU is the baseline validation backend when Warp is installed, CUDA
  validation is optional/local/manual until dedicated CI exists, missing Warp
  and missing CUDA are expected skip paths, and benchmark validation remains
  opt-in rather than part of the standard release gate.
- `docs/contribute/CONTRIBUTING.md` was intentionally not updated so the testing
  guide and roadmap remain the authoritative policy homes.

## Additional Shipped Evidence (E3-F5-P4)

- `particula/gpu/kernels/tests/coagulation_test.py`,
  `particula/gpu/kernels/tests/environment_test.py`,
  `particula/gpu/tests/conversion_test.py`, `particula/gpu/kernels/tests/`
  `condensation_test.py`, and `particula/gpu/kernels/tests/`
  `condensation_stiffness_test.py` now declare module-level `pytest.mark.warp`
  while preserving `pytest.importorskip("warp")` on the suites that import
  Warp directly.
- `particula/gpu/kernels/tests/_condensation_test_support.py` now carries the
  same Warp-policy readability marker while wrapper modules keep explicit Warp
  coverage and `support.device` re-export behavior.
- Representative deterministic cross-device tests are now selectable with
  `gpu_parity` in coagulation, condensation support, and conversion coverage.
- Representative aggregate/statistical coagulation checks are now selectable
  with `stochastic`.
- Representative CUDA-only checks in coagulation, condensation support, and
  environment coverage are now selectable with `cuda` while CPU-first Warp
  coverage remains the default path.
- `E3-F5-P5` completed the final release-command rollout as a docs-only change,
  with no code or test-behavior changes.

## Evidence Metrics

| Metric | Completion Signal | Evidence Source |
| --- | --- | --- |
| Marker registration | No unknown-marker warnings for `warp`, `cuda`, `gpu_parity`, `stochastic` | `particula/conftest.py`, `pyproject.toml`, hook tests |
| CPU-only safety | Focused GPU tests run or skip correctly on Warp CPU-only machines | `cuda_availability_test.py` plus focused pytest runs |
| CUDA optionality | CUDA branches execute when available and emit clean skip reasons when absent | Helper tests and local validation notes |
| Policy adoption | Key GPU kernel modules use the shared marker/helper contract | Edited GPU test modules from P4 |
| Documentation quality | Testing guide and roadmap record exact local/manual commands | `.opencode/guides/testing_guide.md`, roadmap updates |

## Definition of Done

Contributors can tell which GPU tests require only Warp CPU, which are
CUDA-optional, and which use stochastic tolerances by reading one consistent
marker/helper policy backed by tests.
