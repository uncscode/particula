# E2-F7 Overview: Condensation timestep stiffness and integration foundations

## Problem Statement

Issues #1213 and #1214 now cover the first two phases of feature E2-F7. The
current Warp condensation path remains an explicit fixed-step particle-mass
update with non-negative clamping, while aerosol condensation still spans
nanometer-scale stiff behavior through droplet-like regimes. P1 established
the deterministic case catalog, reusable metric helpers, and shared vocabulary;
P2 measured the recorded explicit timestep grid for those same cases without
changing the particle-only production contract.

## Value Proposition

This feature now has a reusable baseline, recorded P2 explicit evidence, and a
shipped P3 candidate-evaluation layer in place. The real test implementation
lives in `_condensation_test_support.py`, while discoverable wrappers
`condensation_test.py` and `condensation_stiffness_test.py` now expose the
named stress cases (`nanometer`, `accumulation_mode`, `droplet_like`),
`CondensationStiffnessCase`/`CondensationStiffnessClassification`, the
recorded timestep grid, test-local candidate helpers
(`fixed_count_substeps_4`, `asymptotic_relaxation`), reusable fixed-shape
scratch coverage, repeated-run determinism checks, finite/non-negative
particle-mass assertions, and CPU-reference plus explicit-baseline comparison
bounds. The roadmap artifact,
`docs/Features/Roadmap/condensation-stiffness-study.md`, now mirrors the
shipped measured-results table and P3 evidence, including graph-capture and
autodiff notes plus the deferred gas-coupling boundary. The public
`condensation_step_gpu(...)` API and production particle-only path remain
unchanged while later phases build recommendations from measured evidence.

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
