# Overview

**Problem Statement:** The GPU condensation path must apply an
inventory-limited transfer consistently to particle mass, gas concentration,
whole-call transfer accounting, and optional energy accounting across four
fixed substeps, while rejecting invalid caller-owned state before mutable work.

**Value Proposition:** Issues #1304 and #1305 ship public orchestration and its
production-path conservation evidence; issue #1306 documents the resulting
bounded direct-kernel contract. Each successful call performs four equal
P1-gated proposal/P2-finalization cycles, applies the finalized transfer to
particles, deterministically couples the concentration-weighted opposite
transfer to active-device gas, and uses the finalized whole-call total for
return/accounting. Regression coverage separately proves CPU mapped-species
conservation and per-box/per-species fp64 GPU inventory bookkeeping against a
CPU oracle. The public API and export boundary remain unchanged.

**User Stories:**
- As a model author, I want binary per-box partitioning flags honored on GPU so
  disabled exchange cannot silently mutate particle state.
- As a caller, I want malformed masks and optional future-sidecar metadata to
  fail before observable state changes so I can correct and retry safely.
- As a maintainer, I want raw proposals, finalized applied transfers, gas
  deltas, and whole-call accounting to have distinct, testable semantics.

Parent epic: [E4](../../epics/E4/vision_problem.md). This feature follows E4-F3 and
E4-F4 and gates E4-F6.
