# Change Log

## 2026-07-14 — E4-F5-P2 shipped (issue #1303)
- Added private, direct-test-only fp64 inventory finalization for already
  P1-gated transfer proposals: owned-mass evaporation bounds, deterministic
  per-box/species demand and release reduction, and gas-plus-release uptake
  scaling.
- Added independent direct-helper oracle and atomic-preflight coverage, plus
  launch-spy/sentinel evidence that the public four-substep P1 path neither
  launches P2 kernels nor touches P2 sidecars.
- Preserved the boundary: no public API, gas coupling, return/energy semantic,
  or user-documentation change; P3--P5 retain those responsibilities.

## 2026-07-14 — E4-F5-P1 shipped (issue #1302)
- Validated active-device binary per-box `WarpGasData.partitioning` masks and
  optional P2 sidecar metadata atomically before mutable condensation work.
- Added a private raw-proposal gate for disabled species and inactive slots.
- Preserved the particle-only `gas.concentration` contract and added focused
  kernel atomicity/gating plus CPU↔Warp conversion regression coverage.

## 2026-07-12 — Initial draft
- Created five issue-sized phases with co-located tests and final documentation.
- Set E4-F3 and E4-F4 as required predecessors and E4-F6 as downstream gate.
- Defined partitioning-first, deterministic inventory limiting and four-substep
  coupled gas/particle mutation using one authoritative finalized transfer.
- Preserved issue #1272's requirement that the production hook and per-box,
  per-species conservation regression land together before support is claimed.
- Recorded no diagnostics beyond required conservation and production evidence.
