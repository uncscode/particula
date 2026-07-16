# Overview

## Problem Statement

Epic E inherits an Epic D closeout gap: direct GPU condensation has focused
tests and support documentation, but no independent, user-followable CPU/Warp
walkthrough that reports physics agreement, inventory conservation, and latent
heat energy bookkeeping as three non-substitutable results. Deferred
capabilities are described in several documents without one ownership record.

## Value Proposition

E5-F8 publishes a deterministic Warp CPU walkthrough whose expected values are
built independently from the Warp execution state. It gives reviewers one
reproducible report with separate pass/fail criteria and gives roadmap owners a
single record for work that the bounded direct kernel does not claim. CUDA is
optional evidence and cannot replace the required Warp CPU result.

## User Stories

- As a scientific reviewer, I want independent CPU and Warp inputs and
  separately reported physics outputs so that shared setup cannot hide an
  implementation error.
- As a release maintainer, I want conservation and energy checks to remain
  distinct from numerical parity so that one passing metric cannot mask a
  failure in another invariant.
- As a roadmap owner, I want every deferred condensation capability assigned to
  a named downstream epic or approval lane so that limitations are not mistaken
  for abandoned or shipped work.

Parent context: [E5](../../epics/E5/child_plans.md), track T8. E5-F8 closes
the E4 carry-forward independently of coagulation tracks E5-F1 through E5-F7;
E5-F9 consumes its published artifacts during roadmap closeout.
