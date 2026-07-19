# Overview

## Problem Statement

Epic E inherits an Epic D closeout gap: direct GPU condensation has focused
tests and support documentation, but no independent, user-followable CPU/Warp
walkthrough that reports physics agreement, inventory conservation, and latent
heat energy bookkeeping as three non-substitutable results. Deferred
capabilities are described in several documents without one ownership record.

## Value Proposition

Issue #1367 delivered the deterministic fp64 walkthrough and its co-located
regression tests. The example completes a detached NumPy fixed-four-substep
oracle before it conditionally imports or allocates Warp state, then exposes
separately synchronized direct-kernel observations. CUDA remains optional
additive evidence; Warp CPU is exercised when Warp is installed.

## User Stories

- As a scientific reviewer, I want independent CPU and Warp inputs and
  separately reported physics outputs so that shared setup cannot hide an
  implementation error.
- As a release maintainer, I want conservation and energy checks to remain
  distinct from numerical parity so that one passing metric cannot mask a
  failure in another invariant.

The implementation remains bounded direct-kernel evidence: it does not claim
high-level CPU strategy or `Runnable` parity, adaptive stepping, performance,
or required CUDA support.

Parent context: [E5](../../epics/E5/child_plans.md), track T8. E5-F8 closes
the E4 carry-forward independently of coagulation tracks E5-F1 through E5-F7;
E5-F9 consumes its published artifacts during roadmap closeout.
