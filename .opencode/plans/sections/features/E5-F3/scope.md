# Scope

E5-F3 extends the E5-F1 sampler with E5-F2 charged physics so the direct GPU
step supports charged-only and Brownian-plus-charged particle-resolved execution
through one bounded candidate and acceptance pass.

## In Scope

- Register the E5-approved charged model as executable in the mechanism
  capability matrix established by E5-F1.
- Compute a finite, non-negative charged majorant that bounds all active charged
  pair rates; use an exhaustive active-pair maximum unless a tighter bound is
  proved and regression-tested.
- Dispatch charged pair rates for charged-only requests.
- Sum Brownian and charged pair rates before one acceptance draw and use a safe
  total majorant for Brownian-plus-charged requests.
- Preserve legacy Brownian behavior, public return values, caller-owned output
  buffer identity, persistent RNG ownership, and fail-before-mutation behavior.
- Cover neutral, same-sign, opposite-sign, inactive-slot, mixed-scale,
  multi-box, conservation, and stochastic execution cases on Warp CPU, with
  optional CUDA coverage that skips cleanly.

## Out of Scope

- Porting charged formulas or merge semantics owned by E5-F2.
- Sedimentation, turbulent shear, four-way combinations, or the full support
  matrix owned by E5-F4 through E5-F7.
- Unsupported charged model variants or runtime Python strategy objects.
- Binned/continuous-PDF coagulation, high-level `Runnable` integration, dynamic
  slot allocation, hidden CPU fallback or transfers, graph capture, adaptive
  stepping, and general performance redesign.
- Exact CPU/Warp stochastic pair replay or CUDA as a release requirement.
