# Success Criteria

## Pass / Fail Criteria

- [ ] The marker vocabulary is explicit and consistent across code, config, and
  docs: `warp`, `cuda`, `gpu_parity`, and `stochastic` are registered without
  unknown-marker warnings.
- [ ] Warp CPU remains the required baseline for GPU-test development, while
  CUDA runs only when available and otherwise skips cleanly with reusable
  helper behavior.
- [ ] Device-selection helpers and skip utilities are covered by focused tests
  that use fakes or monkeypatching rather than requiring real CUDA hardware.
- [ ] Representative GPU kernel tests adopt the standardized markers/helpers
  without weakening existing coverage for condensation, coagulation,
  environment, or conversion paths.
- [ ] The stochastic tolerance policy documents deterministic equality,
  conservation, and aggregate-probabilistic checks with concrete guidance such
  as `3 sigma`-style bands where appropriate.
- [ ] Manual/release validation commands document CUDA-optional workflows and do
  not turn CUDA into a required CI gate.
- [ ] Existing benchmark gating through `--benchmark` remains intact and is not
  conflated with default validation.

## Current Shipped Evidence (E3-F5-P1 to P3)

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
- Remaining unchecked criteria belong to later phases for broader GPU test
  adoption and any follow-on release-command rollout.

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
