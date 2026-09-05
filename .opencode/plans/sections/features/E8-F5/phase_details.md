# Phase Details

- [x] **E8-F5-P1:** Shared full-loop fixtures and independent CPU oracle with unit tests
  - Issue: #1575 | Size: S | Status: Shipped
  - Goal: Build one immutable scenario specification and independent CPU/NumPy
    oracle used to compare all execution paths without production-helper leakage.
  - Files: `particula/execution/tests/captured_full_loop_test.py` only; no shared
    support module or production module was changed.
  - Delivered: Frozen two-box fixture and detached NumPy oracle for closed gas
    communication, prescribed volume evolution, dilution, saturation, and six
    diagnostic outputs.
  - Tests: Fixture schema and immutability, multiple timesteps, literal primary
    and derived values, concentration-weighted inventory, no-work rows, inactive
    slots, and invalid step/scenario rejection.

- [x] **E8-F5-P2:** Uncaptured Warp full-loop parity and conservation validation
  - Issue: #1576 | Size: S | Status: Shipped
  - Delivered: Test-only READY prepared uncaptured Warp-CPU multi-timestep
    parity/conservation evidence in
    `particula/execution/tests/captured_full_loop_test.py`.
  - Tests: Detached primary/derived, closed-GAS work-buffer, accounting, and six
    diagnostic assertions; canonical order and stable identities; scoped
    forbidden-work spies; and zero-duration write-free preservation.
  - Boundary: No production modules, APIs, scheduler behavior, or capture/replay
    behavior changed.

- [x] **E8-F5-P3:** Captured CUDA communication diagnostics and parity validation
  - Issue: #1577 | Size: S | Status: Shipped
  - Delivered: Optional native-CUDA captured resident-loop validation in
    `particula/execution/tests/captured_full_loop_test.py` only. Separate
    closed-map GAS and PARTICLES bindings cover active, prescribed-volume, and
    no-work scenarios; CUDA replay snapshots compare with independently enqueued
    Warp-CPU snapshots, including diagnostics and communication work buffers.
  - Tests: Opaque CUDA-string candidate discovery; pre-capture CUDA/capture-API
    skips; exact-binding qualification; qualification rejection before capture or
    guard entry; replay-only forbidden-host-work instrumentation; CAPTURED
    lifecycle, hidden-handle, and guard-completion checks.
  - Boundary: No production modules, APIs, exports, user documentation, examples,
    scheduler behavior, or architecture changed. CUDA remains optional native
    evidence and never falls back to CPU or Warp-CPU capture.

- [ ] **E8-F5-P4:** RNG continuation and lifecycle rejection matrix with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prove stream advancement/reset/continuation and fail-closed captured
    replay across structural drift and terminal lifecycle states.
  - Files: `particula/execution/tests/captured_full_loop_test.py`,
    `rng_invariance_test.py`, `checkpoint_test.py`, `graph_capture_test.py`.
  - Tests: Coagulation/wall-loss stream identity and advancement, explicit reset,
    checkpoint/restart continuation, stale handle, signature drift, finalize,
    close, fault, teardown, writer-failure, and fresh-only recapture.

- [ ] **E8-F5-P5:** Integrated validation matrix and documentation updates
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Consolidate the default and optional validation commands, document
    tolerances and evidence boundaries, and publish the downstream handoff.
  - Files: `.opencode/guides/testing_guide.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`, E8 plan sections.
  - Tests: Focused resident assertions, optional CUDA pass-or-clean-skip rows,
    untargeted repository coverage runner, documentation contract tests, and
    `mkdocs build --strict`.
