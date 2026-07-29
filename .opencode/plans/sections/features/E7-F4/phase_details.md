# Phase Details

- [x] **E7-F4-P1:** Define resident session state and ownership invariants with unit tests
  - Issue: #1484 (source plan #1460) | Size: S | Status: Implemented
  - Delivered: Concrete-only immutable dimensions, Warp device/ordered gas-name
    metadata, four immutable lifecycle values, and identity-retained generated
    Warp containers with O(1) metadata-only construction validation.
  - Files: `particula/execution/gpu_session.py`,
    `particula/execution/tests/gpu_session_test.py`
  - Tests: CPU-only carrier/import isolation plus Warp-marked identity,
    schema/dtype/shape/device rejection, zero-size boundary, and no-operation
    sentinel coverage. P1 accepts lifecycle values only; transitions remain P4.

- [x] **E7-F4-P2:** Implement one-time conversion and setup with unit tests
  - Issue: #1485 | Size: S | Status: Implemented
  - Delivered: Concrete-only `setup_resident_session()` performs local exact
    Warp-device, CPU carrier, rank, shape, and gas-name preflight before any
    conversion import; uploads particle/gas/environment exactly once in order;
    retains ordered CPU gas names; and publishes a validated `ACTIVE` session.
    E7-F6 native availability remains an explicit upstream precondition.
  - Files: `particula/execution/gpu_session.py`,
    `particula/execution/tests/gpu_session_test.py`
  - Tests: Subprocess import-isolation preflight matrix, Warp-CPU conversion
    spies and identity checks, conversion failure ordering, final schema failure,
    and input immutability. No unavailable-device test was added because the
    E7-F6 availability API is not present.

- [ ] **E7-F4-P3:** Add fixed-shape reusable sidecar registry with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Allocate or accept validated process resources once and preserve all
    array identities across repeated acquisitions and steps.
  - Files: `particula/execution/gpu_resources.py`,
    `particula/execution/tests/gpu_resources_test.py`
  - Tests: Exact shape/dtype/device matrices, key ownership, duplicate-key and
    alias rejection, stable identity, and concrete-record export boundaries.

- [ ] **E7-F4-P4:** Define timestep lifecycle and resident-state guards with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Expose scheduler-facing begin/complete hooks and prohibit conversion,
    resize, checkpoint re-entry, or use after fault/finalization.
  - Files: `particula/execution/gpu_session.py`,
    `particula/execution/tests/gpu_session_test.py`
  - Tests: Multi-step identity, no `from_warp_*` calls, lifecycle ordering,
    nested-operation rejection, and no implicit synchronization.

- [ ] **E7-F4-P5:** Implement explicit checkpoint and finalize operations with restart tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Synchronize once, restore all CPU-owned state, preserve checkpoint
    metadata/resources, support restart, and make finalization terminal.
  - Files: `particula/execution/checkpoint.py`,
    `particula/execution/gpu_session.py`,
    `particula/execution/tests/checkpoint_test.py`
  - Tests: One-sync/three-restore counts, names and metadata, nonterminal
    checkpoint identity, terminal finalize idempotency, and uninterrupted versus
    checkpoint/restart equivalence on Warp CPU.

- [ ] **E7-F4-P6:** Enforce session failure and close semantics with regression tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Keep preflight failures reusable, mark uncertain post-launch failures
    faulted, propagate original errors, and never rollback, restore, or fall back
    implicitly.
  - Files: `particula/execution/gpu_session.py`,
    `particula/execution/tests/gpu_session_test.py`
  - Tests: Failure injection before/after launch, mutation preservation,
    faulted-state guards, explicit discard/close, and cleanup idempotency.

- [ ] **E7-F4-P7:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document setup, ownership, sidecar, checkpoint, restart, failure, and
    explicit-transfer contracts plus E7-F5/E7-F8 extension seams.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, `.opencode/guides/`, `AGENTS.md`
  - Tests: `mkdocs build --strict`, documentation link checks, and public import
    examples under no-Warp and Warp CPU environments.
