# Overview

## Problem Statement

E5-F1 defines a mechanism configuration and one-pass additive sampling contract,
while E5-F2 supplies approved charged pair-rate helpers and charge-conserving
merges. Charged rates depend on particle charge as well as size, so combined
Brownian-plus-charged selection requires a bound over the additive pair rate,
not reuse of a Brownian size-extrema bound.

## Current Delivery

E5-F3-P1 and P2 completed issues #1342 and #1343; P3 completed issue #1344.
The low-level `coagulation_step_gpu` accepts Brownian-only,
`("charged_hard_sphere",)`, and either canonical ordering of
`("brownian", "charged_hard_sphere")` for
`distribution_type="particle_resolved"`. It prepares private fp64
`(n_boxes, n_particles)` total-mass scratch, uses the compact-active O(A²)
charged majorant for charged-only execution, and uses one exhaustive active-pair
scan of the additive rate for combined execution.

Every charged-containing request scans caller-owned charge for finite values
before output allocation, RNG initialization, launches, or mutation. Combined
selection evaluates independently sanitized Brownian and charged terms, sums
them, makes one shared stochastic decision, and uses the existing single
charge-conserving apply launch. Co-located deterministic, stochastic,
ownership/RNG, edge, and mass/charge-conservation tests cover the capability.

## User Stories

- As a maintainer, I want charged-containing execution to retain the shared
  sampler, ownership, RNG, and conservation contracts.
- As a GPU user, I can run the approved low-level Brownian-plus-charged
  configuration in either canonical requested order.

Parent epic: [E5](../../epics/E5/vision_problem.md), **GPU Coagulation Physics
Coverage**. This feature is track T3, follows E5-F1 and E5-F2, and enables E5-F6
and E5-F7. Classifier diagnostics: none.
