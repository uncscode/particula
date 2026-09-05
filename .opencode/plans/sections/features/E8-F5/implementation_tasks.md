# Implementation Tasks

## Backend / Test Support

- [x] Define a test-only immutable full-loop scenario in
  `particula/execution/tests/captured_full_loop_test.py` with fixed float64 state,
  logical box IDs, process controls, maps, diagnostics, and seeds.
- [x] Implement independent CPU/NumPy step and extensive-inventory oracles; do
  not call the production GPU helper under comparison.
- [x] Build the P2 uncaptured READY prepared binding from the existing E8-F2/E8-F3
  seams without changing E8-F4 graph ownership (issue #1576).
- [x] Add P2 test-local assertions for primary/derived fields, closed-GAS work
  buffers, accounting, and each diagnostic output separately (issue #1576).
- [x] Add scoped P2 forbidden-work spies for prepared enqueue setup, allocation,
  upload, readback, and synchronization, plus zero-duration coverage (issue
  #1576).

## Tooling / Tests

- [x] Add CPU-only deterministic fixture/oracle rows for multi-step primary and
  derived state, diagnostics, inventories, no-work behavior, and validation
  rejection (issue #1575). Warp CPU parity remains P2.
- [x] Add optional native-CUDA captured rows for separate GAS and PARTICLES
  communication, active/prescribed-volume/no-work scenarios, diagnostics, and
  family work buffers; retain opaque candidate strings and skip before capture
  when CUDA/capture prerequisites are unavailable (issue #1577).
- [x] Add P3 replay-only forbidden-host-work instrumentation for visible
  conversion, allocation, copy, readback, synchronization, registry acquisition,
  and capture-resource publication; assert qualification rejection before
  capture or guard entry (issue #1577).
- [x] Assert P2 concentration-weighted per-box/per-species inventory independently
  from parity using `rtol=1e-12` and `atol=1e-30` (issue #1576).
- [ ] Add aggregate stochastic checks for coagulation and wall loss without
  requiring exact CPU/CUDA seed trajectories.
- [ ] Add persistent RNG identity, advancement, explicit reset, and checkpoint/
  restart continuation rows.
- [ ] Parameterize structural and lifecycle rejection categories; assert no
  capture launch on preflight failure and session fault on writer launch failure.
- [ ] Run focused tests directly without coverage, then run the untargeted
  `.opencode/tools/run_pytest.py` full-package coverage command.
- [ ] Run Ruff/mypy where applicable and `mkdocs build --strict` for docs changes.
