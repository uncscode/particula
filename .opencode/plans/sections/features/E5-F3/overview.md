# Overview

## Problem Statement

E5-F1 defines a mechanism configuration and one-pass additive sampling contract,
while E5-F2 supplies approved charged pair-rate helpers and charge-conserving
merges. The GPU entry point still cannot execute a charged-only request or add
Brownian and charged rates in one stochastic pass. Charged rates also depend on
particle charge as well as size, so the Brownian extreme-radius bound cannot be
reused without a separate safety proof.

## Value Proposition

E5-F3 makes the first non-Brownian mechanism executable through the existing
particle-resolved `coagulation_step_gpu` boundary. It registers charged-only and
Brownian-plus-charged capability, uses a finite upper bound that covers every
active charged pair, and preserves caller-owned collision buffers and persistent
RNG state. Accepted collisions continue to conserve species mass and, through
E5-F2, total particle charge.

## User Stories

- As a GPU user, I want charged-only coagulation so electrostatic interactions
  can drive a direct particle-resolved step.
- As a GPU user, I want Brownian and charged rates combined before one
  acceptance decision so mechanisms do not create duplicate collision chances.
- As a maintainer, I want the charged majorant and stochastic behavior covered
  by deterministic and bounded tests so later E5 combinations can reuse them.

Parent epic: [E5](../../epics/E5/vision_problem.md), **GPU Coagulation Physics
Coverage**. This feature is track T3, follows E5-F1 and E5-F2, and enables E5-F6
and E5-F7. Classifier diagnostics: none.
