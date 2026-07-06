# E2-F7 Overview: Condensation timestep stiffness and integration foundations

## Problem Statement

Issue #1213 implements the first foundational phase of feature E2-F7. The
current Warp condensation path remains an explicit fixed-step particle-mass
update with non-negative clamping, while aerosol condensation still spans
nanometer-scale stiff behavior through droplet-like regimes. This phase does
not measure stable timestep bounds yet; it establishes the deterministic case
catalog, reusable metric helpers, and shared vocabulary needed so later phases
can measure stiffness without redefining assumptions.

## Value Proposition

This feature now has its P1 baseline in place: `condensation_test.py` defines
named stress cases (`nanometer`, `accumulation_mode`, `droplet_like`),
`CondensationStiffnessCase`/`CondensationStiffnessClassification`, and helper
checks for metadata validity, finite values, non-negativity, fractional mass
change, zero-mass stability, and stable/unstable classification. A new roadmap
artifact, `docs/Features/Roadmap/condensation-stiffness-study.md`, mirrors the
same baseline assumptions. Later phases can now add measured bounds and
integration recommendations against a fixed contract.

## User Stories

- As a GPU condensation implementer, I want deterministic stress cases and
  explicit classification helpers so that later timestep studies use the same
  baseline contract.
- As a roadmap owner, I want a documented integration recommendation so that
  later E2 tracks do not rediscover stiffness trade-offs independently.
- As an autodiff/graph-capture user, I want the selected foundation to avoid
  stochastic or dynamically shaped control flow so that condensation can be
  captured and differentiated in future optimization workflows.

## Parent Epic Context

- Parent epic: E2, Data-Model and Numerical Foundations v2.
- Direct dependencies: E2-F2 environment/schema foundations and E2-F6
  dtype/precision envelope.
- Related sibling tracks: E2-F1 through E2-F6 establish GPU data layout,
  environment boundaries, vapor-property plumbing, kernel migration, and
  precision evidence; E2-F8/E2-F9 are expected to consume this recommendation
  for later implementation/support boundaries.
