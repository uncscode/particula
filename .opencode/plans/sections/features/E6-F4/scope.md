# Scope

Extend E6-F3's low-level particle-resolved Warp wall-loss operation in staged
phases. P1 is shipped and freezes charged-mode configuration validation and
preflight ordering while retaining the existing neutral execution path.

## In Scope

- **Shipped P1:** `NeutralWallLossConfig` has appended `mode`,
  `wall_potential`, and `wall_electric_field` fields, preserving legacy
  positional construction and remaining concrete-module-only.
- **Shipped P1:** neutral/charged mode validation; finite signed scalar
  potential; charged spherical scalar field; and charged rectangular,
  caller-owned same-device `wp.float64` `(3,)` field validation.
- **Shipped P1:** staged rectangular-field validation and finite device scan
  occur after particle schema/device discovery but before particle value scans,
  environment/RNG work, allocation, or mutation. Rejections preserve supplied
  particle, field, and RNG state.
- **Shipped P1:** valid charged configurations execute the unchanged neutral
  coefficient/removal kernel and RNG path; rectangular field buffers retain
  identity and values. Zero-charge configurations therefore exactly retain the
  existing neutral behavior.
- **Deferred P2-P5:** image-charge enhancement, electric-field drift,
  charged-coefficient composition, CPU coefficient parity, and stochastic
  charged-physics validation.
- Integration with E6-F3's active predicate, fixed-shape removal clearing,
  environment normalization, preflight ordering, and caller-owned RNG lifecycle.
- **Deferred P5:** Warp CPU deterministic coefficient parity, predeclared
  statistical survival bounds, and optional CUDA execution with clean skips.
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
