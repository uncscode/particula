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

- [ ] **E8-F4-P2:** Complete fixed-sequence capture and CUDA smoke tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Capture exactly one complete prepared twelve-node enqueue sequence
    with no setup work inside the capture window and publish the graph only
    after `capture_end` succeeds.
  - Files: `particula/execution/graph_capture.py`, E8-F2 prepared enqueue module,
    `particula/execution/tests/{graph_capture,full_loop}_test.py`
  - Tests: Fake launch traces, forbidden allocation/readback/synchronization
    spies, capture-begin/enqueue/end failure cleanup, and CUDA
    pass-or-clean-skip capture smoke coverage.

- [ ] **E8-F4-P3:** Guarded replay and exact compatibility checks with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Validate the current exact resident/prepared/resource/signature
    binding before opening one timestep token and launching the captured graph
    exactly once without reseeding or host process dispatch.
  - Files: `particula/execution/graph_capture.py`,
    `particula/execution/gpu_session.py`,
    `particula/execution/tests/graph_capture_test.py`
  - Tests: Accepted repeated replay, every identity-drift category, duration and
    lifecycle rejection, token completion, persistent RNG advancement, and no
    launch on failed preflight.

- [ ] **E8-F4-P4:** Lifecycle invalidation fault and recapture handling with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Connect structural invalidation, resident fault/finalize/close,
    post-launch failure, teardown, and explicit recapture eligibility to the
    E8-F1 state machine without automatic recapture or rollback.
  - Files: `particula/execution/graph_capture.py`, lifecycle integration points,
    `particula/execution/tests/{graph_capture,gpu_session,checkpoint}_test.py`
  - Tests: Deterministic invalidation reasons, idempotent teardown, stale-handle
    rejection, writer failure faulting, read-only rejection preservation,
    terminal-state behavior, and fresh-record-only recapture.

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
