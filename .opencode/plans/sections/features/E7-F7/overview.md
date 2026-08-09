# Overview

## Problem Statement

Epic G already plans a resident multi-box session (E7-F4), deterministic
process scheduling (E7-F5), and fail-closed backend policy (E7-F6), but its
boxes remain isolated. Issue #1451 Track T7 requires bounded, prescribed
communication so parcel, 1D advection, expansion, and combustion-style cases
can exchange gas and particles and evolve simulation volume without hidden
host transfers or full CFD coupling.

## Value Proposition

E7-F7 has shipped fixed-shape communication declarations, direct-Warp volume,
gas, and particle primitives, plus P5 resident integration. A resident session
can pin exactly one complete closed GAS or PARTICLES map with its native work
record and optional final volumes. The scheduler executes communication, then
volume evolution, as the first two nodes of its closed twelve-node schedule.

P6 adds test-only multi-box independent NumPy `float64` parity and conservation
evidence for both the direct primitives and the resident executor. It does not
change production behavior or the public API.

Those barriers use pre-update volumes and invalidate saturation ratio only;
vapor pressure stays fresh until the existing consumer refresh windows. Schema-v2
checkpoints preserve an optional pinned communication family, while schema-v1
noncommunication checkpoints remain restart-compatible. Normal steps retain
explicit ownership: no transfer, synchronization, fallback, resource
replacement, retry, or rollback is introduced. Independent boxes remain the
default.

## User Stories

- As a parcel-model developer, I want prescribed per-box volume trajectories so
  expansion changes concentrations without losing extensive inventory.
- As a multi-box simulation author, I want a fixed communication map for gas and
  particles so I can model simple advection or mixing without a CFD solver.
- As a maintainer, I want independent CPU/NumPy references, conservation checks,
  and fail-closed validation so resident GPU results are scientifically auditable.

Parent epic: **E7**. Track: **T7**. Scope authority: **issue #1451** and
`docs/Features/Roadmap/data-oriented-gpu.md:1498-1506,1543-1585`.
Classifier diagnostics: **none**.
