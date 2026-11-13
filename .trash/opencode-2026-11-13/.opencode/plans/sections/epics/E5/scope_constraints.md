# Scope and Constraints

## In Scope

- A backward-compatible mechanism configuration, additive semantics, supported
  combination matrix, and fail-before-mutation validation contract.
- Approved charged pair physics, a safe charged majorant, charged-only and
  Brownian-plus-charged execution, and charge-conserving merges.
- SP2016 sedimentation with density/settling properties and collision
  efficiency fixed at 1.
- ST1956 turbulent shear with explicit per-box dissipation and fluid-density
  inputs.
- A safe total majorant and one RNG/sampling pass for additive combinations.
- Independent deterministic CPU/Warp pair fixtures, bounded stochastic tests,
  separate mass and charge conservation, inactive-slot and multi-box cases,
  persistent-RNG and caller-buffer regression coverage.
- Required Warp CPU evidence when Warp is installed and optional CUDA evidence
  that skips cleanly when unavailable.
- The E4 condensation parity walkthrough, deferred-capability ownership record,
  support matrix, direct coagulation example, and roadmap closeout.

## Out of Scope

- DNS turbulence, non-unit sedimentation collision efficiency, binned or
  continuous-PDF GPU coagulation, and unsupported charged variants.
- High-level runnable integration, dynamic slots, fallback execution, hidden
  synchronization, graph capture/replay, broad autodiff, or adaptive stepping.
- General CPU-strategy parity, performance redesign, and exact stochastic pair
  replay across CPU and Warp.

## Constraints

- Python 3.12+, fp64 Warp kernels, fixed-shape arrays, explicit transfers, and
  existing RNG and scratch-buffer ownership must be preserved.
- Public validation errors must occur before particle mutation or RNG
  advancement.
- Combined mechanisms must add pair rates before a single acceptance pass.
- Every implementation track ships module-level `*_test.py` coverage with the
  functions it changes; there is no standalone unit-testing phase.
- Numerical tolerances and stochastic bounds must be declared per fixture,
  with mass and charge checked independently.
- The epic has no fixed deadline and follows dependency order; it closes only
  when all nine feature tracks meet their done signals.
