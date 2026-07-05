# E2-F6 Overview: Mass Representation and Numerical Precision Study

## Problem Statement

Epic E2 prepares data-model and numerical foundations for data-oriented GPU
work. Feature E2-F6 must decide whether the current absolute per-species `fp64` mass
storage is sufficient before any schema or dtype changes are made. The target
simulation range spans new-particle-formation clusters through cloud droplets,
which stresses floating-point dynamic range, conservation accounting, and
small-particle fidelity.

## Value Proposition

This feature now has shipped P1, P2, and P3 study coverage: deterministic
NPF-to-droplet baseline cases, focused candidate-fidelity tests, executable
conservation and mixed-scale error checks, clamp accounting, bounded optional
throughput evidence, and a roadmap page that records the current absolute
per-species `fp64` / `wp.float64` storage policy alongside the executed
study-only candidates. Downstream E2 tracks now have bounded evidence for
candidate reconstruction behavior and P3 tradeoff review without changing the
existing production mass schema or dtype defaults.

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
