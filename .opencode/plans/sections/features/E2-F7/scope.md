# E2-F7 Scope

## In Scope

- Define condensation stiffness stress cases spanning nanometer particles,
  accumulation-mode particles, and cloud-droplet-like regimes.
- Define stability and quality metrics for the current GPU condensation path,
  including metadata validation, non-negative masses, finite-value checks,
  bounded fractional mass change, zero-mass stability, explicit
  stable/unstable classification, and particle-only caveat handling.
- Build or extend benchmark/test harnesses that can later measure stable
  explicit timestep limits for
  `particula.gpu.kernels.condensation.condensation_step_gpu`.
- Compare fixed-shape-compatible integration options:
  - fixed-count sub-stepping,
  - deterministic two-pass/batch-inspired updates,
  - semi-implicit or asymptotic/exponential first-order candidates.
- Document which options are graph-capture friendly and differentiable enough
  for future Warp autodiff experiments.
- Publish the shared baseline assumptions and metric vocabulary in roadmap
  documentation.
- Require a gas-coupled production condensation integration path with
  conservation checks, or an explicit follow-up feature split if that scope is
  too large for E2-F7.

## Out of Scope

- A full production rewrite of GPU condensation.
- A dynamic adaptive timestepper with data-dependent loop counts inside a
  captured graph.
- Stochastic staggered modes or random shuffling in the recommended gradient
  path.
- A new JAX backend; current repository evidence points to Warp graph capture
  and Warp autodiff rather than an implemented JAX path.
- Broad dtype migration; precision choices must remain aligned with E2-F6.
- Final per-box environment migration if E2-F2 containers are not yet
  available. This feature may define contracts and use compatibility shims.
- Leaving gas-coupled production integration unspecified; if not implemented in
  this feature, it must be captured as a follow-up feature with clear gates.
- Timestep sweep tables, measured stability bounds, and integrator comparisons
  in P1.

## Assumptions

- Existing scalar `temperature` and `pressure` API compatibility remains until
  environment containers are available.
- Stress harnesses should run in Warp CPU mode by default and use CUDA only
  when available.
- Any functions added by implementation phases ship with co-located tests in
  the same phase.
