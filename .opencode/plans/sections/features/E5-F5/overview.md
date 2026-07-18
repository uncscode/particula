# Overview

## Problem Statement

The direct particle-resolved GPU coagulation path has Brownian foundations but
does not execute the Saffman-Turner 1956 (ST1956) turbulent-shear collision
term. Turbulent dissipation and carrier-fluid density can differ by simulation
box, so scalar assumptions or density inferred from pressure would make the
physics and ownership contract ambiguous.

## Value Proposition

E5-F5 adds bounded, fp64 turbulent-shear-only execution to the shared E5
sampler. Callers provide explicit positive finite dissipation and fluid-density
values per box; device code derives dynamic and kinematic viscosity and applies
the existing ST1956 equation. The feature preserves fixed-shape buffers,
persistent RNG ownership, fail-before-mutation validation, and the direct
low-level return contract. It makes no DNS turbulence or general turbulence
accuracy claim.

## Implemented Foundation

E5-F5-P1 is complete: internal fp64 Warp helpers now provide kinematic
viscosity (`mu / rho`) and the ST1956 pair rate in
`particula/gpu/dynamics/coagulation_funcs.py`. The implementation is limited
to device physics and co-located equation/invariant coverage; it does not yet
add direct-step inputs, mechanism dispatch, sampling, public exports, or CPU
fallback behavior.

## User Stories

- As a simulation developer, I want explicit per-box dissipation and fluid
  density so heterogeneous boxes use their intended ST1956 rates.
- As a GPU user, I want turbulent-shear collisions to reuse the bounded shared
  sampler so buffers, RNG state, and conservation behavior remain predictable.
- As a scientific reviewer, I want independent CPU/Warp equation checks and a
  clear no-DNS boundary so the supported claim is auditable.

Parent epic: E5. Track: T5. Classifier diagnostics: none.
