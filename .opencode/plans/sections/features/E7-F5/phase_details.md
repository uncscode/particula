# Phase Details

- [x] **E7-F5-P1:** Define typed process capability nodes and validation with unit tests
  - Issue: #1492 | Size: S | Status: Completed
  - Delivered: Immutable closed-catalogue node/dependency declarations and pure
    deterministic validation/normalization, with no scheduler, execution,
    resource access, backend behavior, or package export change.
  - Files: `particula/execution/process_graph.py`, `particula/execution/tests/process_graph_test.py`
  - Tests: immutable records, catalogue exactness, requirements, malformed and
    allowed/disallowed dependencies, canonical cycles/normalization, recovery,
    and guarded no-Warp/GPU import coverage.

- [x] **E7-F5-P2:** Build canonical deterministic dependency order with unit tests
  - Issue: #1493 | Size: S | Status: Completed
  - Delivered: Lexical Kahn topology resolution plus a direct-import-only,
    declaration-only scheduler with selection/profile records, P1-first
    validation, closure, reviewed direction policy, and freshness dependencies.
  - Files: `particula/execution/process_graph.py`,
    `particula/execution/scheduler.py`,
    `particula/execution/tests/process_graph_test.py`,
    `particula/execution/tests/scheduler_test.py`
  - Tests: registration-order invariance, canonical ties, disabled-node and
    freshness closure, both direction profiles, P1-first rejection,
    effective-cycle rejection, and guarded no-backend imports.

- [ ] **E7-F5-P3:** Integrate dilution, wall loss, and nucleation adapters with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Invoke shipped direct boundaries through E7-F4 resource views with no transfer, sync, fallback, or physics rewrite.
  - Files: `particula/execution/process_adapters.py`, `particula/execution/tests/process_adapters_test.py`
  - Tests: identity, exact delegation, sidecars/RNG, validation, no-op, and failure propagation.

- [x] **E7-F5-P4:** Apply prescribed environment and gas updates with unit tests
  - Issue: #1495 | Size: S | Status: Completed
  - Delivered: Concrete-only direct-import requests and executor bind prescribed
    updates to exact resident session/registry/graph/node identities, complete
    deterministic preflight, then copy only designated resident arrays in place.
    Empty canonical schemas are write-free no-ops.
  - Files: `particula/execution/state_updates.py`, `particula/execution/tests/state_updates_test.py`
  - Tests: lazy-Warp binding, role, schema/device/contiguity/alias, scalar,
    copy-order, protected-field, rejected-call immutability, empty-schema, and
    import-isolation coverage.
  - Deferred: scheduler execution, derived refresh, transport, host transfers,
    fallback, package exports, and lifecycle behavior.

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
