# Overview

## Problem Statement

E5-F1 defines a mechanism configuration and one-pass additive sampling contract,
while E5-F2 supplies approved charged pair-rate helpers and charge-conserving
merges. The GPU entry point still cannot execute a charged-only request or add
Brownian and charged rates in one stochastic pass. Charged rates also depend on
particle charge as well as size, so the Brownian extreme-radius bound cannot be
reused without a separate safety proof.

## Current Delivery

E5-F3-P1 is complete for issue #1342 and E5-F3-P2 is complete for issue
#1343. The low-level `coagulation_step_gpu` now accepts exactly
`("charged_hard_sphere",)` with `distribution_type="particle_resolved"`.
It prepares a private fp64 `(n_boxes, n_particles)` total-mass scratch array,
uses the compact-active O(A²) charged majorant, evaluates charged candidate
rates in the existing one-pass selector, and reuses the existing
charge-conserving apply merge.

Charged-only preflight always scans caller-owned charge for finite values before
output allocation, RNG initialization, launches, or mutation. Deterministic and
stochastic charged coverage now exercises rates, majorants, multi-species total
mass, ownership, capacity, conservation, and bounded aggregate behavior.
Brownian-plus-charged execution remains deferred to P3.

## User Stories

- As a maintainer, I want charged-only execution to retain the shared sampler,
  ownership, and conservation contracts.
- As a GPU user, I can run the approved charged-only low-level configuration;
  combined Brownian-plus-charged execution remains unavailable until P3.

Parent epic: [E5](../../epics/E5/vision_problem.md), **GPU Coagulation Physics
Coverage**. This feature is track T3, follows E5-F1 and E5-F2, and enables E5-F6
and E5-F7. Classifier diagnostics: none.
