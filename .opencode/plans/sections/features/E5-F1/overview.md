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

Issue #1331 completed the P1 host-side foundation: a concrete-module-only,
frozen configuration; canonical, unique mechanism resolution; fixed masks; and
a separate executable-capability gate. The remaining E5-F1 work will add
additive pair rates, one bounded sampling pass, and public pre-mutation
validation without changing the existing Brownian runtime until those phases.
The contract preserves caller-owned buffers, persistent RNG state, fixed-shape
fp64 execution, and explicit device ownership.

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
