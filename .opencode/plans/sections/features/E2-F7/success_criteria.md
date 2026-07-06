# E2-F7 Success Criteria

## Done Signal

- A shared condensation stiffness baseline exists for the current explicit GPU
  path.
- A documented integration recommendation is deferred until later phases and is
  cross-referenced from the E2 roadmap once measured evidence exists.
- The recommended direction is compatible with fixed shapes, preallocated
  buffers, deterministic execution, graph capture, and future autodiff.

## Acceptance Criteria

- Stress cases cover at least high-stiffness nanometer/high-supersaturation,
  moderate accumulation-mode, and droplet-like regimes.
- P1 baseline helpers and docs record the current GPU path's particle-only
  limitation explicitly.
- P2 records a fixed timestep grid per named case and proves at least one
  stable and one unstable recorded trial for each case.
- Recorded-grid tests prove caller-owned `mass_transfer` buffer reuse,
  unchanged gas concentration, and scalar-vs-direct-Warp environment-input
  coverage for the current particle-only path.
- P3 evaluates fixed sub-stepping and at least one semi-implicit/asymptotic
  candidate with deterministic, fixed-shape, reusable-buffer evidence while
  keeping those candidates out of the production API unless later phases choose
  one.
- Recommendation identifies required follow-up work for gas concentration
  updates, per-box environment inputs, clamp gradients, or precision changes.
- All new code has co-located tests and passes fast Warp CPU validation.
- Documentation updates identify E2-F2 and E2-F6 as dependencies and separate
  baseline definitions from future measured results.

## Non-Goals Not Required for Completion

- Production-ready adaptive timestepper.
- New JAX backend.
- Complete GPU gas-concentration mutation if outside this feature's size.
- Final autodiff gradient proof; this feature should leave a compatible
  foundation and state remaining proof work.
