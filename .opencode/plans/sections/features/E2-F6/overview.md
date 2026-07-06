# E2-F6 Overview: Mass Representation and Numerical Precision Study

## Problem Statement

Epic E2 prepares data-model and numerical foundations for data-oriented GPU
work. Feature E2-F6 must decide whether the current absolute per-species `fp64` mass
storage is sufficient before any schema or dtype changes are made. The target
simulation range spans new-particle-formation clusters through cloud droplets,
which stresses floating-point dynamic range, conservation accounting, and
small-particle fidelity.

## Value Proposition

This feature now ships a complete P1-P4 study and publication path:
deterministic NPF-to-droplet baseline cases, focused candidate-fidelity tests,
executable conservation and mixed-scale error checks, clamp accounting,
bounded optional throughput evidence, and a canonical recommendation report at
`docs/Features/Roadmap/mass-precision-study.md`. Downstream E2 tracks now have
a published acceptance artifact that keeps the current absolute per-species
`np.float64` / `wp.float64` storage policy in force, links the GPU roadmap and
migration guide back to that report, and documents the evidence required
before any future production dtype or schema change.

## User Stories

- As a numerical developer, I want deterministic NPF-to-droplet precision cases
  so that mass representation choices are tested across the expected dynamic
  range.
- As a GPU implementer, I want fp64/fp32/mixed-precision memory and fidelity
  evidence so that future kernels do not optimize away required accuracy.
- As a maintainer, I want a clear recommendation before schema changes so that
  later tracks can cite a stable decision.

## Parent Epic Context

- Parent: E2, issue #1172.
- Feature: E2-F6.
- Dependency: E2-F1 schema foundation must define the baseline particle data
  model and terminology used by this study.
- Sibling context: E2-F2 through E2-F5 cover environment containers and kernel
  migration boundaries; this feature remains a study/report and does not mutate
  the canonical schema.
