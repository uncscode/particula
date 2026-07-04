# E2-F7 Success Criteria

## Done Signal

- A condensation stiffness map exists for the current explicit GPU path.
- A documented integration recommendation exists and is cross-referenced from
  the E2 roadmap.
- The recommended direction is compatible with fixed shapes, preallocated
  buffers, deterministic execution, graph capture, and future autodiff.

## Acceptance Criteria

- Stress cases cover at least high-stiffness nanometer/high-supersaturation,
  moderate accumulation-mode, and droplet-like regimes.
- Explicit timestep stability results are reproducible and include the current
  GPU path's particle-only limitation.
- Fixed sub-stepping and at least one semi-implicit/asymptotic candidate are
  evaluated or explicitly scoped with rationale if blocked.
- Recommendation identifies required follow-up work for gas concentration
  updates, per-box environment inputs, clamp gradients, or precision changes.
- Gas-coupled production condensation integration is either included with
  conservation checks or split into a named follow-up feature because it is too
  large for this issue-sized plan.
- All new code has co-located tests and passes fast Warp CPU validation.
- Documentation updates identify E2-F2 and E2-F6 as dependencies.

## Non-Goals Not Required for Completion

- Production-ready adaptive timestepper.
- New JAX backend.
- Complete GPU gas-concentration mutation if outside this feature's size.
- Final autodiff gradient proof; this feature should leave a compatible
  foundation and state remaining proof work.
