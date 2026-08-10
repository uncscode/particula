# Overview

## Problem Statement

Issues #1520--#1522 establish the initial resident stochastic RNG ownership
seam. Resident Brownian coagulation and wall loss need stable process-scoped
initial words, independent session-owned sidecars, no implicit reseeding, and
per-logical-box protection from unrelated wall-loss dispatches.

## Value Proposition

`particula.execution.rng` supplies immutable stream keys/descriptors and a
registry for coagulation and wall-loss arrays. P2/P3 carry P1 metadata into the
resident session and initialize one sidecar per process on first acquisition.
Both are retained by identity and nonaliasing; resident dispatch forces
`initialize_rng=False`. P3 passes scheduler-resolved wall-loss selection to a
one-box adapter path, so only selected boxes whose work launches can consume
wall-loss words. Checkpoint/finalize rejects published resident RNG state;
persistence and restart continuation are not implemented.

## User Stories

- As a direct-API caller, I can register stable logical box IDs and lanes without
  requiring Warp or NumPy.
- As a direct-API caller, I can explicitly initialize my existing coagulation and
  wall-loss state arrays after complete preflight.
- As a resident-session user, I get one P1-derived coagulation stream initialized
  once and advanced in place across scheduled Brownian calls.
- As a resident-session user, I get an independent P1-derived wall-loss stream;
  disabled, skipped, zero-time, and no-work logical boxes retain their words.

Parent epic: E7. Issues #1520, #1521, and #1522 completed P1--P3;
reset/inspection, broader invariance, persistence/restart, and documentation
phases remain separate work.
