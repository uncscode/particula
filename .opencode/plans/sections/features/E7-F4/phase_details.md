# Phase Details

- [ ] **E7-F4-P1:** Define resident session state and ownership invariants with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Define typed active/faulted/finalized lifecycle, resident container
    ownership, immutable dimensions/device metadata, and CPU-only metadata.
  - Files: `particula/execution/gpu_session.py`,
    `particula/execution/tests/gpu_session_test.py`
  - Tests: Construction, state transitions, immutable metadata, identity, and
    malformed-state rejection without requiring CUDA.

- [ ] **E7-F4-P2:** Implement one-time conversion and setup with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Upload each CPU container exactly once after E7-F1/E7-F6 capability
    validation and preserve ordered gas metadata.
  - Files: `particula/execution/gpu_session.py`,
    `particula/execution/tests/gpu_session_test.py`
  - Tests: Conversion call counts, Warp CPU setup, unavailable-device errors,
    shape mismatch rejection, and no silent fallback.

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
