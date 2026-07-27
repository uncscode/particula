# Implementation Tasks

## Execution Layer

- [ ] Add immutable graph/node/update types in `particula/execution/process_graph.py`.
- [ ] Encode canonical process IDs, required resources, invalidation sets, and
  dependency edges for supported condensation, Brownian coagulation, dilution,
  wall loss, nucleation, environment, gas, derived-state, and diagnostics nodes.
- [ ] Implement stable topological resolution and deterministic cycle/error reporting.
- [ ] Add `SimulationScheduler` in `particula/execution/scheduler.py`; validate
  the full step before `ResidentSession.begin_step()` and advance counters once.
- [ ] Route E7-F2/F3 adapters through `ExecutionContext` and add resident
  dilution, wall-loss, and nucleation adapters without direct private-kernel use.
- [ ] Preserve all input/output container, array, sidecar, and RNG identities.
- [ ] Fault the session after uncertain launched work and never promise rollback.

## Environment, Gas, and Derived State

- [ ] Add fixed-shape prescribed update declarations and executors in
  `particula/execution/state_updates.py` with deterministic validation order.
- [ ] Keep simulation volume on `ParticleData.volume`; reserve transport and
  expansion policy for E7-F7.
- [ ] Track invalidation explicitly so temperature changes require on-device
  vapor-pressure and saturation refresh before condensation.
- [ ] Reuse `refresh_vapor_pressure_gpu()` and implement/compose the narrow
  saturation refresh boundary without host vapor-pressure evaluation.
- [ ] Ensure condensation gas mutation is visible to later nodes in the same
  step; do not replay or overwrite coupled gas changes.

## Tooling / Tests

- [ ] Add `process_graph_test.py`, `process_adapters_test.py`,
  `state_updates_test.py`, and `scheduler_test.py` under
  `particula/execution/tests/` with every phase.
- [ ] Generalize existing process-sequence fixtures and transfer/sync spies.
- [ ] Test registration-order invariance, graph rejection, exact call order,
  lifecycle, identity, no-op paths, update freshness, and failure propagation.
- [ ] Compare complete Warp CPU loops against independent CPU/NumPy references
  with explicit tolerances and particle-plus-gas conservation checks.
- [ ] Add multi-box isolation rows without transport and optional CUDA rows that
  skip cleanly; do not require exact CPU/CUDA stochastic trajectories.
- [ ] Maintain changed-module coverage at or above 80%; run focused pytest with
  `-Werror`, Ruff, mypy, export regressions, and `mkdocs build --strict`.
