# Overview

## Problem Statement

`coagulation_step_gpu` is a Brownian-only particle-resolved entry point with
mechanism choice embedded in its launch path. E5 needs charged, sedimentation,
turbulent-shear, and combined execution, but there is no stable configuration
contract, no shared pair-rate/majorant interface, and no explicit rejection of
unsupported distribution modes. Adding each mechanism directly to the current
kernel would encourage separate stochastic passes, duplicate collision
opportunities, and validation after state or RNG mutation.

## Value Proposition

Issues #1331 and #1332 completed the P1/P2 foundation: a concrete-module-only,
frozen configuration; canonical fixed masks; Brownian-only capability gating;
and private additive mask dispatch in the Brownian sampler. P2 uses one
sanitized `K_total`, one safe `M_total`, and one acceptance draw per valid
candidate while reserved bits contribute nothing. Issue #1333 completed P3:
`coagulation_step_gpu` now accepts keyword-only `mechanism_config`, preflights
it before any runtime-state access, and dispatches the resolved mask at the
existing kernel boundary. The contract preserves
caller-owned buffers, persistent RNG state, fixed-shape fp64 execution, and
explicit device ownership.

## User Stories

- As a GPU user, I want to select coagulation mechanisms explicitly so that a
  call's physics is reviewable and reproducible.
- As a mechanism implementer, I want one pair-rate and majorant contract so
  that new physics composes without adding independent RNG passes.
- As a caller, I want unsupported distributions and configurations rejected
  before particles or persistent RNG state change.

Parent epic: [E5](../../epics/E5/vision_problem.md), **GPU Coagulation Physics
Coverage**. This feature is track T1 and is foundational for E5-F2, E5-F4, and
E5-F5. Classifier diagnostics: none.
