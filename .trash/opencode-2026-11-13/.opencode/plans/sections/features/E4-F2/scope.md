# Scope

Deliver selected activity and effective surface-tension calculations on device
and integrate them into GPU condensation's particle-side Kelvin pressure.

## In Scope
- Numeric modes for ideal molar activity and kappa water activity.
- Kappa wet, dry, zero-solute, and multi-solute guards with an explicit water
  species index.
- **Delivered in P1 / issue #1287:** internal fp64 Warp ideal-molar and kappa
  helpers plus co-located independent NumPy parity coverage, including zero
  branches and nonzero water-species indices. Warp-absent collection now skips
  safely without eagerly importing Warp-dependent test symbols.
- Legacy static `(n_species,)` surface tension and one explicitly selected
  composition-weighted effective surface mode aligned with CPU fixtures.
- **Delivered in P2 / issue #1288:** internal fp64 Warp effective surface
  tension with static requested-species selection and global single-phase
  composition-volume weighting. The defined zero-volume result is the
  arithmetic mean of supplied species tensions; no public API or launch-path
  change was made.
- `activity * refreshed_pure_vapor_pressure * kelvin_term` in the condensation
  pressure difference, consuming E4-F1's current-temperature pressure buffer.
- **Delivered in P3 / issue #1289:** frozen
  `CondensationActivitySurfaceConfig`, keyword-only `activity_surface`, and
  validation of selectors, water index, fp64 device arrays, domains,
  device/order, and molar-mass ordering before allocation, refresh, launch, or
  caller-state mutation. Configured activity is water-only; legacy calls retain
  unit activity and species-indexed static tension.
- **Delivered in P3 / issue #1289:** weighted tension is reduced once per active
  particle into a step-owned `(n_particles,)` fp64 device buffer, avoiding a
  per-condensing-species full reduction.
- Fixed-shape fp64 arrays, int32 modes, validation before mutation, Warp CPU
  parity, optional CUDA parity, and direct kernel imports.
- Explicit errors or CPU-only documentation for unsupported modes.

## Out of Scope
- BAT and other non-ideal activity models beyond kappa water activity.
- Every CPU activity/surface strategy or general Python strategy dispatch.
- Hidden host recomputation, transfers, container-schema or precision changes.
- High-level `Aerosol`/`Runnable` GPU integration and E4-F3 substep scheduling.
