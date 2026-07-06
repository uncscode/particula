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
- Measured explicit timestep stability results are deferred to P2.
- Fixed sub-stepping and at least one semi-implicit/asymptotic candidate remain
  deferred to P3 unless earlier scope changes are recorded.
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
