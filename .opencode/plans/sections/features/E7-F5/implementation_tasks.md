# Implementation Tasks

## Execution Layer

- [x] Add immutable graph/node types in `particula/execution/process_graph.py`
  (P1, #1492).
- [x] Encode canonical process IDs, required resources, invalidation sets, and
  dependency edges for supported condensation, Brownian coagulation, dilution,
  wall loss, nucleation, environment, gas, derived-state, and diagnostics nodes
  (P1, #1492).
- [x] Implement deterministic declaration normalization and cycle/error
  reporting without an execution-order field (P1, #1492).
- [x] Add `resolve_canonical_topological_order()` in
  `particula/execution/process_graph.py` with lexical Kahn ordering, endpoint
  validation, and effective-cycle rejection (P2, #1493).
- [x] Add direct-import-only declaration scheduler resolution in
  `particula/execution/scheduler.py`: immutable selections/profiles/schedules,
  P1-first validation, closure, direction policy, and freshness dependencies
  (P2, #1493).
- [ ] Add runtime `SimulationScheduler` lifecycle integration after declaration
  resolution; validate before `ResidentSession.begin_step()` and advance
  counters once.
- [ ] Route E7-F2/F3 adapters through `ExecutionContext` and add resident
  dilution, wall-loss, and nucleation adapters without direct private-kernel use.
- [ ] Preserve all input/output container, array, sidecar, and RNG identities.
- [ ] Fault the session after uncertain launched work and never promise rollback.

## Environment, Gas, and Derived State

- [x] Add fixed-shape prescribed environment/gas request declarations and a
  direct-import executor in `particula/execution/state_updates.py` with exact
  resident session/registry/graph/node binding, deterministic preflight,
  protected-field preservation, in-place copy commit, and empty-schema
  write-free no-ops (P4, #1495).
- [ ] Keep simulation volume on `ParticleData.volume`; reserve transport and
  expansion policy for E7-F7.
- [x] Add the P5 concrete coordinator with caller-reported invalidations,
  virtual refresh ordering, and exact resident graph/schedule binding (#1496).
- [x] Reuse `refresh_vapor_pressure_gpu()` and compose a private on-device
  `GAS_CONSTANT` saturation writer without coordinator host payload evaluation
  (#1496).
- [x] Ensure a successful condensation callback invalidates saturation and that
  later diagnostics refresh from its in-place gas mutation without replay or
  overwrite (#1496).

## Tooling / Tests

- [x] Add `process_graph_test.py` with declaration, validation, normalization,
  cycle, and backend-neutral import coverage (P1, #1492).
- [x] Add P2 topology/scheduler tests in `process_graph_test.py` and
  `scheduler_test.py`, including guarded no-backend imports (#1493).
- [ ] Add `process_adapters_test.py` under `particula/execution/tests/` with P3.
- [x] Add `state_updates_test.py` under `particula/execution/tests/` with lazy
  Warp graph-binding, validation, commit-order, immutability, empty-schema, and
   import-isolation coverage (P4, #1495).
- [x] Add `thermodynamic_updates_test.py` with coordinator order, SI-reference,
  identity, rejection, empty-schema, and partial-writer/callback failure
  coverage (P5, #1496).
- [ ] Generalize existing process-sequence fixtures and transfer/sync spies.
- [ ] Test registration-order invariance, graph rejection, exact call order,
  lifecycle, identity, no-op paths, update freshness, and failure propagation.
- [ ] Compare complete Warp CPU loops against independent CPU/NumPy references
  with explicit tolerances and particle-plus-gas conservation checks.
- [ ] Add multi-box isolation rows without transport and optional CUDA rows that
  skip cleanly; do not require exact CPU/CUDA stochastic trajectories.
- [ ] Maintain changed-module coverage at or above 80%; run focused pytest with
  `-Werror`, Ruff, mypy, export regressions, and `mkdocs build --strict`.
