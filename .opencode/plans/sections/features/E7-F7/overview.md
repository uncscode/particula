# Overview

## Problem Statement

Epic G already plans a resident multi-box session (E7-F4), deterministic
process scheduling (E7-F5), and fail-closed backend policy (E7-F6), but its
boxes remain isolated. Issue #1451 Track T7 requires bounded, prescribed
communication so parcel, 1D advection, expansion, and combustion-style cases
can exchange gas and particles and evolve simulation volume without hidden
host transfers or full CFD coupling.

## Value Proposition

E7-F7 has shipped validated fixed-shape communication declarations, direct-Warp
per-box volume evolution, concrete-only synchronous gas communication, and
fixed-capacity direct-Warp particle transport.
The gas operation uses caller-owned extensive ledgers, supports declared open
source/sink endpoints, and commits gas concentration once. The particle
operation uses immutable pre-step planning, exact population matching or
ascending pre-step free-slot reservations, caller-owned ledgers, and a gated
one-kernel commit for closed maps. Resident scheduler integration remains a
later phase. Independent boxes remain the default; shipped operations preserve
explicit ownership, synchronization, and failure boundaries.

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
