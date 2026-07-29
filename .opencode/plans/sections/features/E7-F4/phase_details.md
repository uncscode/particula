# Phase Details

- [x] **E7-F4-P1:** Define resident session state and ownership invariants with unit tests
  - Issue: #1484 (source plan #1460) | Size: S | Status: Shipped
  - Delivered: Concrete-only immutable dimensions, Warp device/ordered gas-name
    metadata, four immutable lifecycle values, and identity-retained generated
    Warp containers with O(1) metadata-only construction validation.
  - Files: `particula/execution/gpu_session.py`,
    `particula/execution/tests/gpu_session_test.py`
  - Tests: CPU-only carrier/import isolation plus Warp-marked identity,
    schema/dtype/shape/device rejection, zero-size boundary, and no-operation
    sentinel coverage. P1 accepts lifecycle values only; transitions remain P4.

- [x] **E7-F4-P2:** Implement one-time conversion and setup with unit tests
  - Issue: #1485 | Size: S | Status: Shipped
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

- [x] **E7-F4-P4:** Define timestep lifecycle and resident-state guards with unit tests
  - Issue: TBD | Size: S | Status: Shipped | Completed: 2026-07-29
  - Goal: Expose scheduler-facing begin/complete hooks and prohibit conversion,
    resize, checkpoint re-entry, or use after fault/finalization.
  - Files: `particula/execution/gpu_session.py`,
    `particula/execution/tests/gpu_session_test.py`
  - Tests: Multi-step identity, no `from_warp_*` calls, lifecycle ordering,
    nested-operation rejection, and no implicit synchronization.

- [x] **E7-F4-P5:** Implement explicit checkpoint and finalize operations with restart tests
  - Issue: #1488 | Size: S | Status: Shipped | Completed: 2026-07-28
  - Delivered: Concrete-only immutable canonical primary/sidecar payloads,
    nonterminal snapshots, cached idempotent terminal finalization, deterministic
    registry enumeration, and explicit same-device fresh-session restart.
  - Files: `particula/execution/checkpoint.py`,
    `particula/execution/gpu_session.py`,
    `particula/execution/tests/checkpoint_test.py`
  - Tests: `particula/execution/tests/checkpoint_test.py` covers immutable
    payloads, inspection detachment, active/terminal lifecycle behavior,
    primary/vapor-pressure recovery, resource-family restart, malformed
    descriptors, and closed-step rejection on Warp CPU.

- [x] **E7-F4-P6:** Enforce session failure and close semantics with regression tests
  - Issue: #1489 | Size: S | Status: Shipped | Completed: 2026-07-29
  - Delivered: Explicit private read-only/writer-uncertain failure outcomes,
    exact-token abort/release without guard accounting advancement, original-error
    preservation, no rollback after uncertain writer failure, and concrete-only
    terminal close/discard. Read-only failures leave sessions reusable; uncertain
    writer failures fault them; active/faulted sessions can close without restore;
    closed/finalized calls are write-free idempotent cases.
  - Files: `particula/execution/gpu_session.py`,
    `particula/execution/tests/gpu_session_test.py`
  - Tests: Co-located failure injection before/after token opening, mutation
    preservation, faulted-state guards, abort/failure-seam validation, original
    exception propagation, explicit discard/close, and no-runtime-work
    idempotency.

- [ ] **E7-F4-P7:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document setup, ownership, sidecar, checkpoint, restart, failure, and
    explicit-transfer contracts plus E7-F5/E7-F8 extension seams.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, `.opencode/guides/`, `AGENTS.md`
  - Tests: `mkdocs build --strict`, documentation link checks, and public import
    examples under no-Warp and Warp CPU environments.
