# Overview

## Problem Statement

Issues #1520--#1522 establish the initial resident stochastic RNG ownership
seam. Resident Brownian coagulation and wall loss need stable process-scoped
initial words, independent session-owned sidecars, no implicit reseeding, and
per-logical-box protection from unrelated wall-loss dispatches.

## Value Proposition

`particula.execution.rng` supplies immutable stream keys/descriptors, frozen
inspection manifests, and selected-lane initialization for coagulation and
wall-loss arrays. P2/P3 carry P1 metadata into the resident session and
initialize one sidecar per process on first acquisition. Both are retained by
identity and nonaliasing; resident dispatch forces `initialize_rng=False`. P4
adds deliberate direct-only inspection and reset calls, gated by an exact ACTIVE
session/registry/closed-guard binding and limited to published sidecars.
Issue #1525 completes P6: schema-v3 checkpoints optionally own immutable,
canonical continuation metadata and current `uint32` words for published
coagulation and wall-loss streams. Exact-device restart builds fresh session,
registry, guard, containers, sidecars, and stream arrays from those words,
without reseeding. Schemas v1/v2 remain restart-compatible without RNG
continuation.

## User Stories

- As a direct-API caller, I can register stable logical box IDs and lanes without
  requiring Warp or NumPy.
- As a direct-API caller, I can explicitly initialize my existing coagulation and
  wall-loss state arrays after complete preflight.
- As a resident-session user, I get one P1-derived coagulation stream initialized
  once and advanced in place across scheduled Brownian calls.
- As a resident-session user, I get an independent P1-derived wall-loss stream;
   disabled, skipped, zero-time, and no-work logical boxes retain their words.
- As a direct resident-session caller, I can inspect frozen stream metadata or
  deliberately reset all published streams or valid selected process/box lanes
  without changing ordinary dispatch behavior.
- As a resident-session user, I have same-device regression evidence that the
  covered logical box's Brownian and selected neutral wall-loss stream/output is
   unchanged by active, removed, no-work, or physically permuted unrelated boxes.
- As a resident-session user, I can checkpoint one or both published streams and
  resume an exact-device split run with the same stochastic state as an
  uninterrupted run; ordinary reacquisition and dispatch do not reseed it.

Issue #1526 completed P7: development documentation now records the shipped
concrete-only ownership, explicit reset/inspection, same-device invariance, and
checkpoint/restart boundaries without claiming new runtime APIs. Parent epic: E7.
