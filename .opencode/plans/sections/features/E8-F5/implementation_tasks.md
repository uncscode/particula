# Implementation Tasks

## Backend / Test Support

- [ ] Define a test-only immutable full-loop scenario in
  `particula/execution/tests/captured_full_loop_test.py` with fixed float64 state,
  logical box IDs, process controls, maps, diagnostics, and seeds.
- [ ] Implement independent CPU/NumPy step and extensive-inventory oracles; do
  not call the production GPU helper under comparison.
- [ ] Build uncaptured and captured bindings from the same E8-F2 prepared plan
  and E8-F3 resource set without replacing E8-F4's graph owner.
- [ ] Add assertion helpers that compare each primary and derived field,
  communication ledger, and diagnostic output separately.
- [ ] Add launch/allocation/readback/synchronization spies around prepared
  replay and rejection rows.

## Tooling / Tests

- [ ] Add deterministic CPU versus Warp CPU uncaptured multi-timestep rows.
- [ ] Add optional CUDA captured rows for GAS and PARTICLES communication,
  prescribed volume evolution, empty/no-work boxes, and diagnostics.
- [ ] Assert concentration-weighted per-box/per-species inventory independently
  from parity, using `rtol=1e-12` and `atol=1e-30` unless evidence requires a
  tighter documented process-specific bound.
- [ ] Add aggregate stochastic checks for coagulation and wall loss without
  requiring exact CPU/CUDA seed trajectories.
- [ ] Add persistent RNG identity, advancement, explicit reset, and checkpoint/
  restart continuation rows.
- [ ] Parameterize structural and lifecycle rejection categories; assert no
  capture launch on preflight failure and session fault on writer launch failure.
- [ ] Run focused tests directly without coverage, then run the untargeted
  `.opencode/tools/run_pytest.py` full-package coverage command.
- [ ] Run Ruff/mypy where applicable and `mkdocs build --strict` for docs changes.
