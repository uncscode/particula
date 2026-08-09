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

- [ ] **E7-F8-P2:** Integrate coagulation streams with resident resources and unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Bind E7-F3 Brownian calls to session-owned process/box streams seeded once and advanced in place.
  - Files: `particula/execution/adapters/coagulation.py`, `particula/execution/session.py`, adapter tests
  - Tests: Identity retention, explicit first initialization, repeated-step continuation, rejection preservation, and no hidden allocation/reset.

- [ ] **E7-F8-P3:** Integrate wall-loss streams with scheduler execution and unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Give wall loss a distinct persistent stream namespace and consume it only for enabled scheduled boxes.
  - Files: `particula/execution/adapters/wall_loss.py`, `particula/execution/scheduler.py`, adapter tests
  - Tests: Process namespace independence, enabled/disabled execution, no-op behavior, identity, and failure propagation.

- [ ] **E7-F8-P4:** Add explicit initialize reset and stream inspection APIs with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Expose deliberate setup/reset operations and immutable stream metadata without normal-step readback.
  - Files: `particula/execution/rng.py`, `particula/execution/session.py`, execution API tests
  - Tests: Lifecycle guards, selected-process/box reset, idempotent inspection, malformed IDs/seeds, and faulted/finalized rejection.

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
