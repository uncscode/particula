# Overview

**Problem Statement:** The production GPU condensation path updates particle
mass from a particle-only request and must reject malformed caller-owned Warp
metadata before any mutable work. Before issue #1302, per-box partitioning
masks and optional P2 scratch sidecars were not fully validated atomically, and
raw proposals could reach application for disabled species or inactive slots.

**Value Proposition:** Issue #1302 delivers E4-F5-P1's safe foundation: each
substep gates disabled species and inactive slots before application, while
preflight validates binary per-box masks and supplied P2 metadata without
mutating particle, gas, or caller-owned scratch state. It deliberately retains
the particle-only, no-gas-mutation contract until later phases implement
inventory limiting and coupled transfer.

**User Stories:**
- As a model author, I want binary per-box partitioning flags honored on GPU so
  disabled exchange cannot silently mutate particle state.
- As a caller, I want malformed masks and optional future-sidecar metadata to
  fail before observable state changes so I can correct and retry safely.
- As a maintainer, I want later gas-coupled support to remain explicitly gated
  on the inventory, conservation, and production work in P2--P4.

Parent epic: [E4](../../../epics/E4/vision_problem.md). This feature follows E4-F3 and
E4-F4 and gates E4-F6.
