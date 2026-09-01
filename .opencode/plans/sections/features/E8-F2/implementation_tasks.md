# Implementation Tasks

## Backend

- [x] Add exact frozen prepared-plan carriers and `prepare_*` validation in the
  focused concrete-only `particula/execution/resident_enqueue.py` module
  (#1552). The READY-only carrier retains metadata by identity and performs no
  executor construction or lifecycle mutation.
- [x] Extract fixed state-update copy launches from validation/readback in
  `particula/execution/state_updates.py`; bind exact P1 input/destination
  identities during preparation and retain standalone execution compatibility
  (#1553).
- [x] Add P1-bound thermodynamic common binding and current-cursor consumer
  preparation in `particula/execution/thermodynamic_updates.py`; enqueue only
  bound vapor-pressure/saturation operations while the coordinator retains
  freshness and cursor ownership (#1553).
- [x] Extract read-only `ResidentDiagnosticsExecutor` validation into a
  functional seam while retaining the executor compatibility wrapper (#1552).
- [x] Bind validated diagnostics registrations, arrays, and outputs to a
  P1-prepared diagnostics carrier with enqueue-only ordered dispatch while
  retaining the standalone executor path (#1553).
- [x] Extract read-only resident communication-request validation and reuse it
  through shared scheduler complete-loop metadata validation (#1552).
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

- [x] Add P1 setup contract tests for exact bindings, lifecycle/signature and
  schedule gating, second-check drift, and no mutation or forbidden host work
  on preparation rejection (#1552).
- [x] Add P2 per-module validate-once/enqueue-only tests for state,
  thermodynamic, and diagnostics seams. They cover identity/pinning rejection,
  writer order, empty no-ops, and traps for setup-only validation, allocation,
  readback, synchronization, registry, schedule, and freshness work (#1553).
- [ ] Run existing direct-kernel test modules to prove public wrappers retain
  validation, atomic preflight, return identity, and numerical behavior.
- [ ] Add a CUDA-gated capture smoke test using the repository's established
  pass-or-clean-skip helper; Warp CPU must explicitly remain uncaptured.
- [ ] Run focused tests without coverage, then the untargeted
  `.opencode/tools/run_pytest.py` full-package coverage gate.
