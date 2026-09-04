# Phase Details

- [x] **E8-F4-P1:** Prepared resident graph-capture qualification controller with
  unit tests
  - Issue: #1567 | Size: S | Status: Delivered
  - Delivered: A direct-import-only controller binds one exact E8-F1 READY
    binding, E8-F2 prepared simulation, and E8-F3 published capture set; lazy
    adapter probes qualify opaque non-CPU Warp devices and retain frozen native
    callable records by identity.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/tests/graph_capture_test.py`,
    `particula/execution/tests/exports_test.py`
  - Verified contract: ordered lazy probes, exact links, READY-preserving success
    and failure, no token/native-callable invocation/handle/cleanup ownership,
    and denied package/top-level exports. Native capture and replay remain P2/P3.

- [x] **E8-F4-P2:** Complete fixed-sequence capture and CUDA smoke tests
  - Issue: #1568 | Size: S | Status: Delivered
  - Delivered: `capture_prepared_resident_graph()` captures one already-qualified
    prepared simulation as `capture_begin()` → the retained twelve-operation
    dispatch → `capture_end()`. It revalidates before begin and after end,
    publishes `CapturedResidentGraph` only after the binding becomes CAPTURED,
    and privately releases an end handle after post-end rejection. The opaque
    handle is retained and released by identity only.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/resident_scheduler.py`,
    `particula/execution/tests/{graph_capture,full_loop,exports}_test.py`
  - Verified contract: fake-runtime ordering and failure cleanup, capture-window
    prohibition of normal scheduler/token/validation/resource/transfer/readback/
    synchronization work, retained-operation ordering, denied package/top-level
    exports, and a CUDA-only native twelve-no-op smoke row. P3 replay remains
    deferred.

- [x] **E8-F4-P3:** Guarded replay and exact compatibility checks with unit tests
  - Issue: #1569 | Size: S | Status: Delivered
  - Delivered: `replay_captured_resident_graph()` accepts only authentic
    P2-issued opaque-handle records, verifies their exact captured binding and
    duration before token entry, then performs one native launch and one token
    completion. It preserves payload/RNG-word compatibility and applies
    writer-capable no-rollback fault handling to launch or completion failures.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/tests/graph_capture_test.py`
  - Verified contract: provenance rejects manual/tampered records before launch;
    identity, lifecycle, device, and duration drift reject before token entry;
    accepted calls launch the retained handle exactly once; package and top-level
    exports remain unchanged.

- [x] **E8-F4-P4:** Lifecycle invalidation fault and recapture handling with tests
  - Issue: #1570 | Size: S | Status: Delivered
  - Delivered: Graph-owned teardown unregisters issued records before exactly one
    native release and drives nondispatchable lifecycle transition for drift,
    writer fault, finalization, close/discard, and retirement. Lifecycle owners
    use an exact session/registry/closed-guard lazy notification; stream
    initialization now validates that guard.
  - Files: `particula/execution/{graph_capture,gpu_session,checkpoint,gpu_resources}.py`,
    `particula/execution/tests/{graph_capture,gpu_session,checkpoint,gpu_resources}_test.py`
  - Verified contract: stale provenance rejects before token/launch, release is
    exactly once even after callback failure, read-only rejection preserves the
    record, writer and terminal paths invalidate it, and renewal/recapture issues
    only a distinct fresh record without handle checkpointing or public exports.

- [ ] **E8-F4-P5:** Three-way full-loop validation and development documentation
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Validate multiple identical process-sequence timesteps across CPU,
    uncaptured Warp, and captured CUDA, then document the implemented internal
    contract and downstream evidence handoff.
  - Files: `particula/execution/tests/captured_full_loop_test.py`, existing CPU
    oracle/integration fixtures, `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/data-containers-and-gpu-foundations.md`, `AGENTS.md`
  - Tests: Per-field deterministic tolerances, tight conservation, aggregate
    stochastic bounds, RNG continuation/reset, no hidden operations, Warp CPU
    uncaptured baseline, CUDA pass-or-clean-skip rows, docs contracts, and
    `mkdocs build --strict`.
