# Phase Details

Phase issue creation is intentionally deferred until E4 implementation issues
are generated and scheduled; `TBD` is not an unresolved design decision.

- [x] **E4-F6-P1:** Add device-aware condensation parity matrix with independent CPU references
  - Issue: #1308 | Size: S | Status: Completed
  - Delivered: Two shared deterministic fp64 cases compare the direct GPU step
    with an independent NumPy fixed-four-substep/P2/gas-coupled oracle on Warp
    CPU and, when available, CUDA.
  - Files: `particula/gpu/kernels/tests/_condensation_test_support.py`, `condensation_test.py`
  - Tests: One-box uptake/latent-heat/inventory-limited and multi-box,
    multi-species mixed-phase/gated cases; mass and gas are asserted separately
    at `rtol=1e-10` with per-output scale-derived finite `atol`. Coverage
    includes uptake, evaporation, disabled partitioning, zero gas, and inactive
    particle slots; CUDA skips cleanly when unavailable.

- [x] **E4-F6-P2:** Add per-box per-species conservation and mutation-contract regressions
  - Issue: #1309 | Size: S | Status: Completed
  - Delivered: Warp-CPU contract regressions separately prove
    concentration-weighted particle-plus-gas conservation for each box/species,
    P2-finalized total-transfer accounting, and unweighted latent-energy
    accounting.
  - Files: `particula/gpu/kernels/tests/_condensation_test_support.py`,
    `particula/gpu/kernels/tests/condensation_test.py`
  - Tests: Inactive, disabled, and zero-concentration entries; inventory-limited
    uptake; finite/nonnegative final state; immutable caller inputs;
    caller-owned output identity; atomic representative invalid-buffer paths;
    and deterministic runs with fresh state and sidecars. No production API or
    physics changes were made.

- [ ] **E4-F6-P3:** Prove fixed-loop reusable-buffer graph-capture readiness
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Capture and replay the exactly-four-substep path with stable caller-owned buffers.
  - Files: condensation support and a discoverable graph-readiness test module.
  - Tests: Stable identities/shapes, no required allocation with complete scratch, replay parity and conservation, supported-device skips.

- [ ] **E4-F6-P4:** Add bounded autodiff-readiness experiments and limitation tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Exercise supported smooth interiors and record clamp/in-place limitations without promising production gradients.
  - Files: condensation support, autodiff-readiness tests.
  - Tests: Deterministic tape/gradcheck where supported, array-access verification, explicit expected skip/limitation cases.

- [ ] **E4-F6-P5:** Update development documentation and evidence matrix
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish backend, invariant, graph, and autodiff evidence with focused commands.
  - Files: issue 1272 roadmap/feature documentation and testing guidance.
  - Tests: Markdown links, command/reference verification.
