# E2-F6 Overview: Mass Representation and Numerical Precision Study

## Problem Statement

Epic E2 prepares data-model and numerical foundations for data-oriented GPU
work. Feature E2-F6 must decide whether the current absolute per-species `fp64` mass
storage is sufficient before any schema or dtype changes are made. The target
simulation range spans new-particle-formation clusters through cloud droplets,
which stresses floating-point dynamic range, conservation accounting, and
small-particle fidelity.

## Value Proposition

This feature produces an evidence-backed report that keeps `fp64` as the
reference, compares candidate storage/precision alternatives, and records a
recommendation for future schema work. Downstream E2 tracks gain a documented
decision boundary: no mass representation or dtype migration proceeds until the
study validates conservation and accuracy tradeoffs.

## User Stories

- As a numerical developer, I want NPF-to-droplet precision cases so that mass
  representation choices are tested across the expected dynamic range.
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
