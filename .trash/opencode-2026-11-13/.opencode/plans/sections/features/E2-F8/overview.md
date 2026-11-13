# E2-F8 Overview: CPU Dynamics Data-Container Support Boundaries

## Problem Statement

Issue #1172 feature E2-F8 clarifies a user-facing ambiguity introduced by the
E2 data-container migration: `ParticleData` and `GasData` support multi-box
shapes at the container/schema level, but current CPU dynamics strategies do
not provide full multi-box execution semantics. P1 for issue #1218 documented
the existing baseline, P2 for issue #1219 shipped explicit single-box runtime
enforcement for CPU coagulation `ParticleData` paths while broadening
representative public condensation rejection coverage, and P3 for issue #1220
completed the docs-only contract update that makes the migration guide the
canonical user-facing CPU support boundary.

## Value Proposition

- Makes the support boundary explicit: container multi-box shape support is not
  the same as strategy-level multi-box execution support.
- Publishes the migration guide as the canonical support contract for current
  CPU condensation and coagulation container workflows.
- Gives users clear guidance for supported single-box runs,
  caller-managed per-box loops, and future multi-box strategy work.
- Preserves momentum from dependency E2-F1 by documenting and testing the
  supported container contract before later feature tracks expand behavior.

## User Stories

- As a particula user adopting `ParticleData`/`GasData`, I want docs and tests
  to say which CPU dynamics calls are single-box only so that I do not assume
  multi-box execution exists by default.
- As a developer maintaining condensation and coagulation strategies, I want
  the roadmap and migration guide to describe the same `n_boxes == 1` CPU
  boundary so support wording does not regress silently.
- As a future multi-box implementer, I want the current unsupported-path
  enforcement captured precisely so that full multi-box work has a clean
  baseline.

## Parent Epic Context

Parent epic: E2, issue #1172. E2 covers schema foundation, environment
containers, gas/environment boundaries, migration paths, numerical evidence,
and support-boundary documentation. Sibling features E2-F1 through E2-F7 set up
the container schemas, environment/gas boundaries, kernel migration studies,
and numerical comparisons that this feature references. E2-F8 is a boundary
clarification track, not a full multi-box dynamics implementation.
