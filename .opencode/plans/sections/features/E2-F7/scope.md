# E2-F7 Scope

## In Scope

- Define condensation stiffness stress cases spanning nanometer particles,
  accumulation-mode particles, and cloud-droplet-like regimes.
- Define stability and quality metrics for the current GPU condensation path:
  non-negative masses, bounded explicit mass increments, CPU/GPU parity where
  applicable, conservation notes, and sensitivity to timestep size.
- Build or extend benchmark/test harnesses that measure stable explicit
  timestep limits for `particula.gpu.kernels.condensation.condensation_step_gpu`.
- Compare fixed-shape-compatible integration options:
  - fixed-count sub-stepping,
  - deterministic two-pass/batch-inspired updates,
  - semi-implicit or asymptotic/exponential first-order candidates.
- Document which options are graph-capture friendly and differentiable enough
  for future Warp autodiff experiments.
- Publish a stiffness map and integration recommendation in the E2 roadmap
  documentation.

## Out of Scope

- A full production rewrite of GPU condensation.
- A dynamic adaptive timestepper with data-dependent loop counts inside a
  captured graph.
- Stochastic staggered modes or random shuffling in the recommended gradient
  path.
- A new JAX backend; current repository evidence points to Warp graph capture
  and Warp autodiff rather than an implemented JAX path.
- Broad dtype migration; precision choices must remain aligned with E2-F6.
- Final per-box environment migration if E2-F2/T2 containers are not yet
  available. This feature may define contracts and use compatibility shims.

## Assumptions

- Existing scalar `temperature` and `pressure` API compatibility remains until
  environment containers are available.
- Stress harnesses should run in Warp CPU mode by default and use CUDA only
  when available.
- Any functions added by implementation phases ship with co-located tests in
  the same phase.
