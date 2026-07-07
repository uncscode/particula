# E2-F8 Overview: CPU Dynamics Data-Container Support Boundaries

## Problem Statement

Issue #1172 feature E2-F8 clarifies a user-facing ambiguity introduced by the
E2 data-container migration: `ParticleData` and `GasData` support multi-box
shapes at the container/schema level, but current CPU dynamics strategies do
not provide full multi-box execution semantics. CPU condensation already
requires `n_boxes=1`; CPU coagulation accepts `ParticleData` but currently
operates on box 0 through legacy-compatible adapters. Phase P1 for issue
#1218 stays audit-only and locks that baseline in with focused regressions so
later phases can change validation or docs from an explicit tested contract.

## Value Proposition

- Makes the support boundary explicit: container multi-box shape support is not
  the same as strategy-level multi-box execution support.
- Prevents silent misuse of CPU coagulation paths that only inspect or mutate
  box 0.
- Gives users clear guidance for single-box runs, caller-managed per-box loops,
  and future multi-box strategy work.
- Preserves momentum from dependency E2-F1 by documenting the current
  container contract before later feature tracks expand behavior.

## User Stories

- As a particula user adopting `ParticleData`/`GasData`, I want docs and tests
  to say which CPU dynamics calls are single-box only so that I do not assume
  multi-box execution exists by default.
- As a developer maintaining condensation and coagulation strategies, I want
  focused tests around multi-box inputs so that support boundaries do not
  regress silently.
- As a future multi-box implementer, I want existing unsupported paths and
  transitional box-0 behavior captured precisely so that full multi-box work
  has a clean baseline.

## Parent Epic Context

Parent epic: E2, issue #1172. E2 covers schema foundation, environment
containers, gas/environment boundaries, migration paths, numerical evidence,
and support-boundary documentation. Sibling features E2-F1 through E2-F7 set up
the container schemas, environment/gas boundaries, kernel migration studies,
and numerical comparisons that this feature references. E2-F8 is a boundary
clarification track, not a full multi-box dynamics implementation.
