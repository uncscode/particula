# Phase Details

- [ ] **E4-F6-P1:** Add device-aware condensation parity matrix with independent CPU references
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compare combined E4 physics per box and species on Warp CPU and optional CUDA.
  - Files: `particula/gpu/kernels/tests/_condensation_test_support.py`, `condensation_test.py`
  - Tests: One/multi-box fp64 cases, independent one-box references, explicit tolerances, clean backend skips.

- [ ] **E4-F6-P2:** Add per-box per-species conservation and mutation-contract regressions
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prove strict inventory/energy bookkeeping and validation-before-mutation.
  - Files: condensation support and discoverable contract tests.
  - Tests: Tight separate conservation assertions, inactive-entry checks, invalid-buffer non-mutation, deterministic repeats.

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
