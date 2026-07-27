# Overview

## Problem Statement

Epic G already plans a resident multi-box session (E7-F4), deterministic
process scheduling (E7-F5), and fail-closed backend policy (E7-F6), but its
boxes remain isolated. Issue #1451 Track T7 requires bounded, prescribed
communication so parcel, 1D advection, expansion, and combustion-style cases
can exchange gas and particles and evolve simulation volume without hidden
host transfers or full CFD coupling.

## Value Proposition

E7-F7 adds validated, fixed-shape communication descriptions and resident
operations for conservative gas and particle transport, simple mixing, and
per-box volume evolution. Independent boxes remain the default; enabling a map
produces deterministic, identity-stable updates compatible with later graph
capture while preserving explicit ownership, synchronization, and failure
boundaries.

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
