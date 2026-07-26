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

- [x] **E6-F9-P2:** Validate the complete direct GPU process sequence on Warp CPU
  - Issue: #1447 | Size: S | Status: Complete
  - Delivered: expanded `particula/gpu/tests/process_sequence_test.py` with
    private P2 resident composition coverage across condensation, coagulation,
    dilution, charged wall loss, and nucleation.
  - Test design: test-local all-enabled fixture variants retain original sparse
    disabled-condensation coverage; stable same-device containers and persistent
    sidecars/RNGs span calls; conversion is guarded until final inspection.
  - Evidence: accounting and no-op assertions, preflight immutability,
    exhaustion-policy behavior, RNG reuse/advancement, required Warp CPU rows,
    and optional CUDA rows. Neutral wall loss is separately tested with a
    stochastic aggregate assertion.
  - Boundary: no production coordinator, export, hidden transfer/fallback,
    runnable, or public API was added.

- [x] **E6-F9-P3:** Publish the explicit-transfer complete-process example with regression tests
  - Issue: #1448 | Size: S | Status: Complete
  - Delivered: `docs/Examples/gpu_complete_process_sequence.py` and
    `particula/gpu/tests/gpu_complete_process_sequence_example_test.py`.
  - Contract: exactly one conversion of each CPU container; condensation,
    coagulation, dilution, wall loss, then nucleation; one synchronization; and
    exactly one final restore of each container. Sidecars and RNG remain
    caller-owned, no-Warp execution is deterministic, and errors never select a
    CPU fallback.
  - Files: `docs/Examples/gpu_complete_process_sequence.py`,
    `particula/gpu/tests/gpu_complete_process_sequence_example_test.py`.
  - Tests: CPU-only forced/natural-unavailable runs and subprocess output; fake
    runtime transfer/order/identity assertions; parameterized failure propagation;
    and guarded real Warp CPU execution.

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
