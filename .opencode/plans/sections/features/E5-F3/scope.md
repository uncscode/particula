# Scope

E5-F3 stages E5-F2 charged physics for eventual charged-only and
Brownian-plus-charged particle-resolved execution. Issue #1342 completed only
the P1 private term-majorant foundation; public execution remains deferred.

## In Scope

- Compute a finite, non-negative charged majorant that bounds all active charged
  pair rates by exhaustively scanning unique compact active pairs.
- Dispatch the approved charged hard-sphere term internally through the existing
  term-majorant dispatcher, while retaining Brownian dispatcher behavior.
- Cover physics fixtures, invalid/zero candidates, sparse compact active lists,
  per-box behavior, dispatcher addition, and Brownian regression with
  co-located independent deterministic tests.
- In later phases, register executable capability; route candidate rates through
  one selection/acceptance pass; and preserve public API, buffer, RNG, and
  mass/charge-merge contracts.

## Out of Scope

- Porting charged formulas or merge semantics owned by E5-F2.
- Sedimentation, turbulent shear, four-way combinations, or the full support
  matrix owned by E5-F4 through E5-F7.
- Unsupported charged model variants or runtime Python strategy objects.
- Binned/continuous-PDF coagulation, high-level `Runnable` integration, dynamic
  slot allocation, hidden CPU fallback or transfers, graph capture, adaptive
  stepping, and general performance redesign.
- Exact CPU/Warp stochastic pair replay or CUDA as a release requirement.
- Public charged-only or Brownian-plus-charged execution in P1, including
  capability registration, selection, acceptance, merging, or API changes.
