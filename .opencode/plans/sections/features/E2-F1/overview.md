# E2-F1 Overview: Container Schema, Shapes, and Ownership

## Problem Statement

Issue #1172 Epic E2 needs a stable data-model foundation before downstream
GPU and multi-box work can proceed. The repository already has CPU containers
(`ParticleData`, `GasData`) and GPU mirrors (`WarpParticleData`,
`WarpGasData`), but ownership boundaries are not fully authoritative:
`WarpGasData` contains `vapor_pressure` while `GasData` does not,
environment state is still mostly scalar API input, and per-box versus shared
fields need consistent rules.

## Value Proposition

This feature creates the canonical schema and shape contract for particle, gas,
and environment state. It lets E2 sibling tracks implement `EnvironmentData`,
`WarpEnvironmentData`, Gas/WarpGas alignment, scalar-to-per-box migration, and
documentation with explicit ownership decisions instead of re-litigating field
semantics in each track.

## User Stories

- As a data-model implementer, I want one schema decision record so I can add
  container fields without conflicting ownership between gas, particle, and
  environment state.
- As a GPU-kernel implementer, I want explicit CPU/GPU shape conventions so I
  can validate transfers and kernel launches for single-box and multi-box
  workflows.
- As a downstream planner, I want field ownership handoffs from T1 so tracks
  T2-T9 can implement against the same authoritative source.

## Parent Epic Context

- Parent epic: E2, Issue #1172, Data-Model and Numerical Foundations v2.
- Scenario: epic-linked feature.
- Sibling tracks documented by the epic drafter: E2-F2 through E2-F9. Their
  work depends on T1 decisions for environment fields, GPU mirrors, gas schema
  drift, scalar migration, numerical evidence, CPU support boundaries, and final
  user-facing docs.
