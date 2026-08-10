# Phase Details

- [x] **E7-F8-P1:** Define stable per-box stream identities and deterministic seeding with unit tests
   - Issue: #1520 | Size: S | Status: Complete
  - Goal: Define versioned process/box stream keys and derive reproducible initial `uint32` state without positional coupling.
  - Files: `particula/execution/rng.py`, `particula/execution/tests/rng_test.py`
   - Delivered: direct-only host identity/registration, FNV derivation, lazy
     NumPy/Warp initializer, caller-buffer preflight/copy, and export-denial
     coverage; no session, scheduler, or checkpoint integration.
   - Tests: validation, namespace separation, stable-ID invariance, seed vectors,
     collision handling, host-only import, initializer preflight, and exports.

- [x] **E7-F8-P2:** Integrate coagulation streams with resident resources and unit tests
  - Issue: #1521 | Size: S | Status: Complete
  - Goal: Bind E7-F3 Brownian calls to one session-owned P1-derived stream seeded
    once and advanced in place, without checkpoint persistence.
  - Files: `particula/execution/gpu_session.py`, `particula/execution/gpu_resources.py`,
    `particula/execution/adapters/coagulation.py`,
    `particula/execution/resident_scheduler.py`, `particula/execution/checkpoint.py`,
    and co-located execution tests.
  - Delivered: Immutable resident stream metadata; one first-acquisition
    coagulation-only sidecar pinned by identity; literal-false resident dispatch;
    and checkpoint/finalize rejection after publication.
  - Tests: Metadata and sidecar schemas, one-time initialization, identity and
    advancement, no-op/rejection preservation, scheduler binding, and pre-payload
    checkpoint/finalize failure.

- [x] **E7-F8-P3:** Integrate wall-loss streams with scheduler execution and unit tests
   - Issue: #1522 | Size: S | Status: Complete
   - Goal: Give wall loss a distinct persistent stream namespace and consume it
     only for scheduler-selected boxes whose direct work launches.
   - Files: `particula/execution/gpu_resources.py`,
     `particula/execution/process_adapters.py`,
     `particula/execution/resident_scheduler.py`,
     `particula/execution/checkpoint.py`, and co-located execution tests.
   - Delivered: Canonical-manifest initialization publishes one independent,
     session-owned wall-loss `wp.uint32` sidecar retained by identity and
     nonaliasing with coagulation. The resolved scheduler selection reaches the
     adapter, which invokes the unchanged direct kernel through one-box views;
     disabled, prelaunch-skipped, zero-time, and no-work lanes remain unchanged.
     Checkpoint rejection and post-launch scheduler fault/token lifecycle are
     preserved.
   - Tests: Namespace/initial-word derivation, transactional acquisition and
     reacquisition, resource/view nonaliasing, selected-box identity and gating,
     reset-like request rejection, no-launch/no-work preservation, checkpoint
     guard, and post-dispatch fault propagation.

- [x] **E7-F8-P4:** Add explicit initialize reset and stream inspection APIs with unit tests
  - Issue: #1523 | Size: S | Status: Complete
  - Goal: Expose deliberate setup/reset operations and immutable stream metadata without normal-step readback.
  - Files: `particula/execution/rng.py`, `particula/execution/gpu_resources.py`,
    `particula/execution/gpu_session.py`, and co-located execution tests.
  - Delivered: Frozen host-only stream manifests, selected-lane writes after full
    preflight, published-sidecar-only resource operations, and exact ACTIVE
    session/registry/closed-guard lifecycle composition. `reset_streams()` is a
    deliberate alias of `initialize_streams()`; neither changes normal dispatch,
    public exports, checkpointing, or restart.
  - Tests: Lifecycle guards, selected-process/box reset, idempotent inspection,
    strict selector rejection, empty-published no-op, preservation before
    writers, and faulted/finalized rejection.

- [ ] **E7-F8-P5:** Guarantee disabled added and reordered box stream invariance with regressions
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prove logical identity, not storage position or neighboring activity, determines each enabled stream.
  - Files: `particula/execution/tests/rng_invariance_test.py`, scheduler/adapter files as needed
  - Tests: Added, removed, disabled, and permuted boxes for coagulation and wall loss on Warp CPU; optional CUDA rows.

- [ ] **E7-F8-P6:** Implement checkpoint restart RNG continuation semantics with integration tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Snapshot and restore all stream metadata/state so split and uninterrupted same-backend runs agree exactly.
  - Files: `particula/execution/checkpoint.py`, `particula/execution/session.py`, `particula/execution/tests/checkpoint_test.py`
  - Tests: Checkpoint schema validation, one-sync boundary, exact continuation, reset-after-restart, incompatibility rejection, and no intermediate transfer.

- [ ] **E7-F8-P7:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish ownership, seeding, reset, invariance, checkpoint, restart, and support boundaries.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`
  - Tests: `mkdocs build --strict`, documentation links, snippets, and focused command verification.
