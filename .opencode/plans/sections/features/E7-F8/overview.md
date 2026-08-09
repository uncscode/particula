# Overview

## Problem Statement

Issues #1520 and #1521 establish the initial resident Brownian-coagulation RNG
ownership seam. Resident stochastic calls needed stable process-scoped initial
words, one resident-owned sidecar, and no implicit reseeding during dispatch.

## Value Proposition

`particula.execution.rng` supplies immutable stream keys/descriptors and a
registry for coagulation and wall-loss arrays. P2 carries P1 metadata into the
resident session, initializes exactly one coagulation-only `wp.uint32` sidecar
on first resource acquisition, retains it by identity, and always dispatches
resident Brownian work with `initialize_rng=False`. Checkpoint/finalize rejects
published resident RNG state; persistence and restart continuation are not
implemented.

## User Stories

- As a direct-API caller, I can register stable logical box IDs and lanes without
  requiring Warp or NumPy.
- As a direct-API caller, I can explicitly initialize my existing coagulation and
  wall-loss state arrays after complete preflight.
- As a resident-session user, I get one P1-derived coagulation stream initialized
  once and advanced in place across scheduled Brownian calls.

Parent epic: E7. Issue #1520 completed P1 and issue #1521 completed P2;
wall-loss, reset/inspection, invariance, persistence/restart, and broader
documentation phases remain separate work.
