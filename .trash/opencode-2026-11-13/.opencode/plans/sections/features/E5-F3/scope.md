# Scope

E5-F3 stages E5-F2 charged physics for charged-only and Brownian-plus-charged
particle-resolved execution. Issues #1342, #1343, and #1344 completed the
private majorant foundation, charged-only execution, and combined execution.

## In Scope

- Compute a finite, non-negative charged majorant that bounds all active charged
  pair rates by exhaustively scanning unique compact active pairs.
- Dispatch the approved charged hard-sphere term internally through the existing
  term-majorant dispatcher, while retaining Brownian dispatcher behavior.
- Cover physics fixtures, invalid/zero candidates, sparse compact active lists,
  per-box behavior, dispatcher addition, and Brownian regression with
  co-located independent deterministic tests.
- Execute only exact `charged_hard_sphere` particle-resolved requests through one
  selection/acceptance pass, using private summed-total-mass scratch and forced
  finite-charge preflight while preserving API, buffer, RNG, and merge contracts.
- Execute only canonical Brownian-plus-charged particle-resolved requests through
  the same single selector and apply path. Normalize either requested order, sum
  independently sanitized terms, and bound the sum with one exhaustive active-
  pair scan; preserve preflight state, caller-buffer identity, and RNG reuse.

## Out of Scope

- Porting charged formulas or merge semantics owned by E5-F2.
- Sedimentation, turbulent shear, four-way combinations, or the full support
  matrix owned by E5-F4 through E5-F7.
- Unsupported charged model variants or runtime Python strategy objects.
- Binned/continuous-PDF coagulation, high-level `Runnable` integration, dynamic
  slot allocation, hidden CPU fallback or transfers, graph capture, adaptive
  stepping, and general performance redesign.
- Exact CPU/Warp stochastic pair replay or CUDA as a release requirement.
