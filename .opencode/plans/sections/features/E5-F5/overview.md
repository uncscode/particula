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

E5-F5-P2 is complete: `coagulation_step_gpu` now accepts keyword-only
`turbulent_dissipation` and `fluid_density` inputs. Its local normalizer accepts
positive finite Python/NumPy floating scalars or supported same-device Warp
arrays shaped `(n_boxes,)`, preserves valid array identity, and uses private
same-device storage for scalar broadcasts. Turbulent requests validate both
inputs before downstream runtime setup and then fail at the unchanged reserved
ST1956 capability gate; non-turbulent calls ignore these arguments.

## User Stories

- As a simulation developer, I want explicit per-box dissipation and fluid
  density so heterogeneous boxes use their intended ST1956 rates.
- As a GPU user, I want turbulent-shear collisions to reuse the bounded shared
  sampler so buffers, RNG state, and conservation behavior remain predictable.
- As a scientific reviewer, I want independent CPU/Warp equation checks and a
  clear no-DNS boundary so the supported claim is auditable.

Parent epic: E5. Track: T5. Classifier diagnostics: none.
