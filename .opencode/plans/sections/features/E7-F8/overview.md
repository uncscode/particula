# Overview

## Problem Statement

Issue #1520 completes E7-F8-P1's direct RNG ownership seam. Existing stochastic
sidecars needed stable process-scoped initial words without importing GPU
dependencies during host-only identity registration or rebinding caller arrays.

## Value Proposition

`particula.execution.rng` now supplies immutable stream keys/descriptors and a
registry for coagulation and wall-loss arrays. It uses deterministic host-only
FNV derivation and explicit, validated initialization of caller-owned Warp
buffers. Session, scheduler, checkpoint/restart, and reset integration remain
deferred.

## User Stories

- As a direct-API caller, I can register stable logical box IDs and lanes without
  requiring Warp or NumPy.
- As a direct-API caller, I can explicitly initialize my existing coagulation and
  wall-loss state arrays after complete preflight.

Parent epic: E7. Issue #1520 completes P1 only; E7-F8-P2--P7 remain separate
integration work.
