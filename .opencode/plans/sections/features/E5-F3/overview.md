# Overview

## Problem Statement

E5-F1 defines a mechanism configuration and one-pass additive sampling contract,
while E5-F2 supplies approved charged pair-rate helpers and charge-conserving
merges. The GPU entry point still cannot execute a charged-only request or add
Brownian and charged rates in one stochastic pass. Charged rates also depend on
particle charge as well as size, so the Brownian extreme-radius bound cannot be
reused without a separate safety proof.

## Current Delivery

E5-F3-P1 is complete for issue #1342. The private GPU term-majorant dispatcher
now supports the approved charged hard-sphere term using a finite,
non-negative, exhaustive maximum over unique compact active pairs. It reuses
the E5-F2 charged physics and sanitizes invalid candidates.

This is internal dispatcher support only. `coagulation_step_gpu` still does not
execute charged-only or Brownian-plus-charged configurations; P2 and P3 retain
capability registration, selection, acceptance, and merge integration.

## User Stories

- As a maintainer, I want an independently tested charged majorant so later
  execution phases have a safe internal bound.
- As a GPU user, I will receive charged-only and combined execution only after
  P2 and P3 complete their public-step integration.

Parent epic: [E5](../../epics/E5/vision_problem.md), **GPU Coagulation Physics
Coverage**. This feature is track T3, follows E5-F1 and E5-F2, and enables E5-F6
and E5-F7. Classifier diagnostics: none.
