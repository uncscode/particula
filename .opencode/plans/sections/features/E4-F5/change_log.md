# Change Log

## 2026-07-14 — E4-F5-P4 shipped (issue #1305)
- Added regression-only concentration-weighted particle-plus-gas inventory
  conservation coverage; production APIs, container schemas, and the fixed
  four-substep behavior are unchanged.
- Updated the CPU particle-resolved integration regression to account for H2O
  and NH4HSO4 independently and to require exact invariance of gas-only N2.
- Added a deterministic fp64 two-box public `condensation_step_gpu()` case with
  uptake, evaporation, disabled partitioning, zero gas, and zero-concentration
  slots. It checks per-box/per-species inventory at `rtol=1e-12, atol=1e-30`
  separately from CPU-oracle parity at `rtol=2e-10, atol=1e-30`, on Warp CPU
  and guarded CUDA.

## 2026-07-14 — E4-F5-P3 shipped (issue #1304)
- Completed public four-fixed-substep P1/P2 orchestration in
  `particula/gpu/kernels/condensation.py`: finalized transfer is applied once,
  accumulated into the returned total, and coupled deterministically to gas.
- Added aggregate atomic preflight and one-per-call scratch resolution, plus
  finalized energy accounting while retaining API, exports, return arity, and
  caller-buffer identity.
- Added focused Warp wrapper/support regression coverage for oracle coupling,
  sequence/order, boundaries, ownership/metadata atomicity, fresh-proposal
  failures, and scratch reuse. Later-cycle fresh-proposal failure intentionally
  retains already completed substeps; it is not whole-call rollback.

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
