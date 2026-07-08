# Implementation Strategy

## Architecture Overview

Epic C stays within the existing low-level GPU architecture:

- `particula.gpu` owns explicit transfer helpers, GPU data mirrors, and optional
  Warp availability exports.
- `particula.gpu.kernels` owns direct condensation and coagulation kernel entry
  points.
- `particula.gpu.kernels.environment` normalizes direct scalar/Warp-array and
  `WarpEnvironmentData` inputs without hidden device transfers.
- GPU tests live under `particula/gpu/kernels/tests/` and `particula/gpu/tests/`;
  CPU latent-heat baselines live under `particula/integration_tests/` or the
  relevant condensation test package.
- User-facing examples live under `docs/Examples/` and should be smoke-tested
  using existing example-test patterns.

## Key Data Ownership Rules

- Caller-owned Warp arrays remain caller-owned. Kernels may mutate documented
  output/state arrays such as coagulation RNG state, but must not allocate or
  transfer hidden CPU/GPU mirrors behind the caller's back.
- `rng_states` semantics must be explicit: caller-supplied state should advance
  across calls rather than being reinitialized unless the API documents another
  opt-in behavior.
- CPU metadata such as gas names remains CPU-owned and outside kernel state.
- CUDA is a conditional validation target, not a mandatory dependency.

## Reusable Patterns

- Use `pytest.importorskip("warp")` for Warp-dependent tests.
- Use `warp_devices(wp)` to cover CPU and CUDA-if-available devices.
- Use statistical assertions for stochastic coagulation behavior, following the
  existing mean-within-tolerance and directional-behavior tests.
- Use strict mass/energy conservation assertions only for deterministic
  quantities.
- Follow `docs/Examples/data_containers_and_gpu_foundations.py` for optional
  Warp example structure and smoke-test coverage.

## Testing Requirements

1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Coverage expectations must match repository configuration: run the relevant
   `pytest`/`pytest --cov=particula` checks when coverage evidence is needed,
   but do not claim that `pyproject.toml` configures a fixed 80% minimum gate.

## Track-Level Strategy

- E3-F1 updates `particula/gpu/kernels/coagulation.py` and expands coagulation
  tests around caller-supplied RNG state.
- E3-F2 adds mixed-scale fixtures and either improves rejection-sampling
  behavior or documents measured limitations with regression coverage.
- E3-F3 records the threading decision and measured benchmark boundaries in
  roadmap/architecture documentation.
- E3-F4 either promotes kernel exports or documents `particula.gpu.kernels` as
  the supported path, then adds a runnable quick-start.
- E3-F5 centralizes device-aware pytest expectations in docs and, where useful,
  test helpers or marker configuration.
- E3-F6 adds a CPU `CondensationLatentHeat` example without adding GPU latent
  heat physics.
- E3-F7 adds an integration-level CPU conservation baseline that future GPU
  parity tests can compare against.
