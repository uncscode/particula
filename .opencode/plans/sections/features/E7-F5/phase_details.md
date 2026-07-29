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

- [x] **E7-F5-P5:** Order vapor-pressure and saturation refreshes with unit tests
  - Issue: #1496 | Size: S | Status: Completed
  - Delivered: A concrete-only coordinator retains exact resident
    session/registry/graph/schedule/configuration identities, receives explicit
    successful-node reports, and consumes virtual vapor-pressure/saturation
    writers immediately before scheduled condensation or diagnostics callbacks.
    It owns stale markers and a schedule cursor; partial writer failures retain
    successful freshness markers without cursor advancement.
  - Files: `particula/execution/thermodynamic_updates.py`,
    `particula/execution/tests/thermodynamic_updates_test.py`,
    `.opencode/guides/architecture_reference.md`, and
    `.opencode/guides/architecture/architecture_outline.md`.
  - Tests: canonical cursor/writer/callback order; stale-writer elision;
    one- and multi-box/species SI saturation references; identity preservation;
    empty-schema paths; binding/role/order/callback rejection; and vapor,
    saturation, and callback partial-failure markers.
  - Deferred: lifecycle/guard ownership, resource acquisition, full-timestep
    scheduler dispatch, transport, transfer, fallback, public exports, and
    general process dispatch.

- [x] **E7-F5-P6:** Add diagnostics hooks and complete-loop integration tests
  - Issue: #1497 | Size: S | Status: Completed
  - Delivered: Direct-import-only diagnostics implement two closed snapshots
    with caller-owned validated outputs and canonical empty no-ops. The
    direct-import-only resident scheduler accepts only the exact resolved
    ten-node schedule, binds the active session/registry/closed guard by
    identity, opens one token after whole-loop preflight, and dispatches the
    resolver order with thermodynamic consumer windows. Writer-capable failures
    fault without rollback.
  - Files: `particula/execution/diagnostics.py`,
    `particula/execution/resident_scheduler.py`,
    `particula/execution/gpu_resources.py`,
    `particula/execution/tests/scheduler_test.py`, and the two concrete
    architecture guides.
  - Tests: closed-protocol/output rejection/no-op coverage; complete-loop order,
    lifecycle, freshness, identity, conservation/loss, isolation, repeatability,
    no-transfer/sync/fallback, read-only abort, uncertain-writer faulting, Warp
    CPU, and skip-clean optional CUDA.

- [ ] **E7-F5-P7:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document process order, state authority, support boundaries, errors, and downstream handoffs.
  - Files: `docs/Features/`, `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`, E7 plan sections
  - Tests: `mkdocs build --strict`, documentation regression and import/export checks.
