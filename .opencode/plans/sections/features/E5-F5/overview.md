# Overview

## Problem Statement

The direct particle-resolved GPU coagulation path has Brownian foundations but
does not execute the Saffman-Turner 1956 (ST1956) turbulent-shear collision
term. Turbulent dissipation and carrier-fluid density can differ by simulation
box, so scalar assumptions or density inferred from pressure would make the
physics and ownership contract ambiguous.

## Value Proposition

E5-F5 now provides bounded, fp64 turbulent-shear-only execution on the shared
E5 sampler. A structurally valid ST1956 singleton requires explicit positive
finite scalar or same-device `wp.float64` `(n_boxes,)` dissipation and
fluid-density inputs. Mixed turbulent masks reject in preflight; the singleton
uses an O(A) two-largest-active-radii majorant and the shared single candidate
and acceptance stream. Caller-owned collision buffers/RNG and the in-place,
post-launch non-rollback model are retained. The feature makes no DNS or general
turbulence accuracy claim.

## Implemented Foundation

E5-F5-P1 is complete: internal fp64 Warp helpers now provide kinematic
viscosity (`mu / rho`) and the ST1956 pair rate in
`particula/gpu/dynamics/coagulation_funcs.py`. The implementation is limited
to device physics and co-located equation/invariant coverage; it does not yet
add direct-step inputs, mechanism dispatch, sampling, public exports, or CPU
fallback behavior.

E5-F5-P2/P3 are complete: `coagulation_step_gpu` accepts keyword-only
`turbulent_dissipation` and `fluid_density` inputs and executes the exact
particle-resolved ST1956 singleton. Each input accepts a positive finite
Python/NumPy floating scalar or an active-device `wp.float64` `(n_boxes,)` Warp
array; valid arrays retain identity and scalar broadcasts use private same-device
storage. Turbulent mixed masks validate both P2 inputs and then reject before
normalization, allocation, RNG work, launch, or mutation. Non-turbulent masks
ignore these arguments.

## User Stories

- As a simulation developer, I want explicit per-box dissipation and fluid
  density so heterogeneous boxes use their intended ST1956 rates.
- As a GPU user, I want turbulent-shear collisions to reuse the bounded shared
  sampler so buffers, RNG state, and conservation behavior remain predictable.
- As a scientific reviewer, I want independent CPU/Warp equation checks and a
  clear no-DNS boundary so the supported claim is auditable.

Parent epic: E5. Track: T5. Classifier diagnostics: none.
