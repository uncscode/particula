# Overview

## Problem Statement

Particula has mature CPU wall-loss strategies for neutral spherical and
rectangular chambers, but no low-level Warp implementation. A fixed-shape GPU
timestep therefore cannot apply neutral particle-resolved wall loss without a
host transfer or unsupported fallback. The missing path must preserve the CPU
coefficient equations and stochastic survival semantics while safely clearing
removed fixed slots and advancing caller-owned RNG state across timesteps.

## Value Proposition

E6-F3 supplies a parity-tested direct GPU wall-loss step for the two neutral
geometries. It keeps particle arrays resident and allocation-stable, makes RNG
ownership explicit, rejects invalid calls before mutation, and establishes the
neutral foundation required by sibling E6-F4's charged wall-loss extension.

## Delivered Phase: E6-F3-P1 (#1401)

P1 shipped the reusable neutral fp64 Warp transport foundation, not a wall-loss
step. `particula.gpu.properties` is now the sole owner and import surface for
the migrated particle radius, diffusion, effective-density, settling, and slip
helpers; legacy definitions and re-exports under `particula.gpu.dynamics` were
removed and consumers were migrated. The phase also adds device-only
`debye_1_wp` and `x_coth_x_wp` geometry factors and defines safe zero/invalid
behavior for `cunningham_slip_correction_wp`.

No coefficient assembly, wall-loss configuration or API, slot removal/RNG
behavior, charged physics, or CPU behavior shipped in P1. Those remain the
scope of later E6-F3 phases and E6-F4.

## User Stories

- As a particle-resolved simulation developer, I want neutral spherical and
  rectangular wall loss on Warp so that I can keep process state device-resident.
- As a physics maintainer, I want deterministic coefficient comparisons against
  the CPU strategies and statistically bounded survival outcomes so that the GPU
  implementation is scientifically reviewable without requiring identical RNG
  sequences.
- As a workflow integrator, I want persistent caller-owned RNG and complete slot
  clearing so repeated direct steps preserve reproducibility and fixed-slot
  invariants.
