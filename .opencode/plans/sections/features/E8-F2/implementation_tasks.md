# Implementation Tasks

## Backend

- [ ] Add exact frozen prepared-plan carriers and `prepare_*` validation in
  `particula/execution/resident_scheduler.py` or a focused concrete-only
  `particula/execution/resident_enqueue.py` module.
- [ ] Extract fixed state-update copy launches from validation/readback in
  `particula/execution/state_updates.py`; bind input arrays and validation
  status during preparation.
- [ ] Resolve thermodynamic refresh windows once and expose fixed
  vapor-pressure/saturation enqueue operations in
  `particula/execution/thermodynamic_updates.py`.
- [ ] Split `ResidentDiagnosticsExecutor` into explicit validate/prepare and
  enqueue methods while preserving the canonical operation order and empty
  schemas in `particula/execution/diagnostics.py`.
- [ ] Split communication and volume metadata validation from resident primitive
  calls in `particula/execution/resident_communication.py` and
  `particula/gpu/kernels/communication.py`.
- [ ] Refactor `condensation_step_gpu` so its public boundary validates and then
  calls one private prepared launch sequence in
  `particula/gpu/kernels/condensation.py`.
- [ ] Refactor resident Brownian coagulation, dilution, and selected/all-box wall
  loss into validated setup plus private enqueue in
  `particula/gpu/kernels/{coagulation,dilution,wall_loss}.py` and their resident
  adapters.
- [ ] Refactor nucleation and nested exhaustion operations into a prepared,
  fixed launch sequence in `particula/gpu/kernels/{nucleation,exhaustion}.py`.
- [ ] Compose all prepared operations in canonical schedule order without
  per-node dictionary lookup, executor construction, allocation, validation,
  import resolution, host readback, or synchronization during enqueue.
- [ ] Preserve lifecycle token and writer-failure semantics, E8-F1 signature
  checks before enqueue, direct API validation ordering, and all existing
  concrete-only export boundaries.

## Tooling and Tests

- [ ] Add setup/enqueue contract tests under `particula/execution/tests/` for
  exact bindings, deterministic operation order, lifecycle gating, and no
  mutation on setup rejection.
- [ ] Add per-module tests proving prepared enqueue performs no `wp.zeros`,
  `wp.array`, `.numpy()`, transfer, synchronization, resource acquisition,
  runtime device selection, or RNG initialization.
- [ ] Run existing direct-kernel test modules to prove public wrappers retain
  validation, atomic preflight, return identity, and numerical behavior.
- [ ] Add a CUDA-gated capture smoke test using the repository's established
  pass-or-clean-skip helper; Warp CPU must explicitly remain uncaptured.
- [ ] Run focused tests without coverage, then the untargeted
  `.opencode/tools/run_pytest.py` full-package coverage gate.
