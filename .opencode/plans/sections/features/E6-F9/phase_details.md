# Phase Details

## Sequencing

All E6-F1 through E6-F8 plans must ship before integration. Complete P1 before
P2, use P2 evidence for P3, and close E6 in P4 only after P1-P3 pass.

- [x] **E6-F9-P1:** Build integrated fixed-shape validation fixtures and invariants with tests
  - Issue: #1446 | Size: S | Status: Complete
  - Delivered: `particula/gpu/tests/process_sequence_test.py`, a private,
    deterministic fp64 fixture/invariant module with one-/multi-box sparse
    fixtures, snapshot/ownership helpers, independent inventory, dilution,
    wall-loss, slot, and exhaustion expectations, plus optional Warp mirrors.
  - Boundary: no production or public API change and no integrated resident
    process-sequence execution.
  - Files: `particula/gpu/tests/process_sequence_test.py` and test-local fixture
    helpers.
  - Tests: Fixture schemas/repeatability; expected no-op and mutation fields;
    particle-plus-gas and coagulation mass/charge accounting; dilution/wall-loss
    budgets; slot/exhaustion outcomes; and fixed shapes, identities, diagnostics.

- [ ] **E6-F9-P2:** Validate the complete direct GPU process sequence on Warp CPU
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Execute condensation, coagulation, dilution, wall loss, and nucleation
    consecutively on the same Warp-resident containers and caller-owned state.
  - Files: `particula/gpu/tests/process_sequence_test.py`.
  - Tests: Required Warp CPU run; optional CUDA run; persistent RNG; no
    intermediate conversion; per-process parity and budgets; invalid-input
    snapshots; repeated-call stability.

- [ ] **E6-F9-P3:** Publish the explicit-transfer complete-process example with regression tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Publish a runnable low-level example with explicit setup/checkpoint
    transfers and no scheduler or backend-selection abstraction.
  - Files: `docs/Examples/gpu_complete_process_sequence.py`,
    `particula/gpu/tests/gpu_complete_process_sequence_example_test.py`.
  - Tests: CPU-only import/run path, lazy Warp imports, stable output, direct
    entry-point order, sidecar identity, one setup conversion, and one restore.

- [ ] **E6-F9-P4:** Update development documentation, roadmap cross-links, and epic closeout
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Publish support boundaries and evidence, cross-link E6/E6-F1-F9, and
    close E6 only when its dependency-gated exit bar passes.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/Roadmap/index.md`, relevant `docs/Features/` pages,
    `docs/index.md`, `AGENTS.md`, and E6/E6-F9 plan sections.
  - Tests: Documentation link/import/command checks, plan validation, and
    explicit assertions that Epic G scheduling/backend selection remains
    pending and out of E6 scope.
