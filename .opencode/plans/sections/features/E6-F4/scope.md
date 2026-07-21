# Scope

Extend E6-F3's low-level particle-resolved Warp wall-loss operation with the
bounded charged coefficient defined by the CPU `ChargedWallLossStrategy`, then
validate coefficient parity, neutral fallback, stochastic outcomes, and all
existing fixed-slot and persistent-RNG invariants.

## In Scope

- An immutable charged configuration extending the E6-F3 spherical and
  rectangular geometry contract with finite wall potential [V] and finite
  scalar or three-component electric field [V/m].
- Device-side image-charge enhancement using particle charge in elementary-
  charge counts, including enhancement at zero wall potential.
- Potential-derived and explicitly configured field drift, preserving CPU sign,
  geometry scaling, clipping, finite sanitization, and nonnegative composition.
- Exact per-slot neutral fallback when charge is zero, including when a field or
  wall potential is configured.
- Integration with E6-F3's active predicate, fixed-shape removal clearing,
  environment normalization, preflight ordering, and caller-owned RNG lifecycle.
- Warp CPU deterministic coefficient parity and predeclared statistical
  survival bounds; optional CUDA execution with clean skips.
- Focused support/deferred documentation and direct-kernel import coverage.

## Out of Scope

- Changes to the CPU charged or neutral wall-loss equations or public strategy.
- Discrete/continuous distributions, a GPU `Runnable`, backend selection,
  scheduling, adaptive stepping, multi-box transport, or process composition.
- Dynamic particle allocation, resizing, compaction, activation, resampling, or
  exhaustion handling; E6-F5/E6-F6 own those capabilities.
- Hidden CPU/GPU transfers, host coefficient fallback, container schema changes,
  hidden RNG ownership, or exact NumPy/Warp random-sequence matching.
- General electrostatic chamber/CFD coupling, alternate charging models,
  graph capture, autodiff, mandatory CUDA, benchmarks, or performance claims.
