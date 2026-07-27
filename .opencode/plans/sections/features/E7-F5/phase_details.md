# Phase Details

- [ ] **E7-F5-P1:** Define typed process capability nodes and validation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Model immutable process/update nodes, requirements, and rejection order without importing Warp in the neutral layer.
  - Files: `particula/execution/process_graph.py`, `particula/execution/tests/process_graph_test.py`
  - Tests: duplicate/unknown nodes, unsupported capabilities, cycles, malformed dependencies, deterministic normalization.

- [ ] **E7-F5-P2:** Build canonical deterministic dependency order with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Resolve a stable topological order with environment and derived-state hazards encoded as edges.
  - Files: `particula/execution/process_graph.py`, `particula/execution/scheduler.py`, co-located tests
  - Tests: registration-order invariance, disabled nodes, tie breaking, stale-state prevention, prelaunch failures.

- [ ] **E7-F5-P3:** Integrate dilution, wall loss, and nucleation adapters with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Invoke shipped direct boundaries through E7-F4 resource views with no transfer, sync, fallback, or physics rewrite.
  - Files: `particula/execution/process_adapters.py`, `particula/execution/tests/process_adapters_test.py`
  - Tests: identity, exact delegation, sidecars/RNG, validation, no-op, and failure propagation.

- [ ] **E7-F5-P4:** Apply prescribed environment and gas updates with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Apply validated fixed-shape per-box updates at explicit graph nodes before their consumers.
  - Files: `particula/execution/state_updates.py`, `particula/execution/tests/state_updates_test.py`
  - Tests: shape/dtype/device/alias checks, positivity, nonnegativity, update ordering, rejected-call immutability.

- [ ] **E7-F5-P5:** Order vapor-pressure and saturation refreshes with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Refresh derived thermodynamic state exactly when invalidated and before condensation or diagnostics consume it.
  - Files: `particula/execution/thermodynamic_updates.py`, scheduler and co-located tests
  - Tests: temperature invalidation, unchanged-state elision, multi-species saturation, call order, no host evaluation.

- [ ] **E7-F5-P6:** Add diagnostics hooks and complete-loop integration tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Complete one resident timestep contract and expose ordered, non-mutating hook points.
  - Files: `particula/execution/scheduler.py`, `particula/execution/diagnostics.py`, `particula/execution/tests/scheduler_test.py`
  - Tests: five-process loops, no intermediate transfer/sync, lifecycle counters, conservation, repeatability, faulting, Warp CPU and optional CUDA.

- [ ] **E7-F5-P7:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document process order, state authority, support boundaries, errors, and downstream handoffs.
  - Files: `docs/Features/`, `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`, E7 plan sections
  - Tests: `mkdocs build --strict`, documentation regression and import/export checks.
