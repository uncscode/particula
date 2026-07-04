# E2-F7 Overview: Condensation timestep stiffness and integration foundations

## Problem Statement

Issue #1172 track T7 addresses the numerical stiffness of the current GPU
condensation timestep. The existing Warp path uses an explicit fixed-step
particle-mass update with non-negative clamping, while aerosol condensation can
span microsecond equilibration for nanometer particles and second-scale cloud
droplet evolution. Without a stiffness map and integration recommendation, the
E2 GPU roadmap risks choosing a timestep strategy that is either unstable,
overly expensive, incompatible with graph capture, or hostile to autodiff.

## Value Proposition

This feature produces a measured stiffness envelope and an actionable
integration foundation for follow-up GPU condensation work. The result should
tell implementers when the current explicit path is safe, when fixed
sub-stepping is enough, and whether a deterministic semi-implicit/asymptotic
update should become the preferred foundation. The recommendation must align
with E2-F2 environment/container work and E2-F6 precision decisions, and it
must preserve fixed shapes, preallocated buffers, deterministic execution, and
gradient-friendly control flow.

## User Stories

- As a GPU condensation implementer, I want stress cases and stable timestep
  bounds so that I can avoid unstable explicit updates when extending the
  Warp path.
- As a roadmap owner, I want a documented integration recommendation so that
  later E2 tracks do not rediscover stiffness trade-offs independently.
- As an autodiff/graph-capture user, I want the selected foundation to avoid
  stochastic or dynamically shaped control flow so that condensation can be
  captured and differentiated in future optimization workflows.

## Parent Epic Context

- Parent epic: E2, Data-Model and Numerical Foundations v2.
- Direct dependencies: E2-F2/T2 environment/schema foundations and E2-F6/T6
  dtype/precision envelope.
- Related sibling tracks: E2-F1 through E2-F6 establish GPU data layout,
  environment boundaries, vapor-property plumbing, kernel migration, and
  precision evidence; E2-F8/E2-F9 are expected to consume this recommendation
  for later implementation/support boundaries.
