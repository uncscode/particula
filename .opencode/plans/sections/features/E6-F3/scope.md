# Scope

Deliver a direct, particle-resolved Warp step for neutral wall loss in
spherical and rectangular chambers. The step ports the CPU transport and
coefficient equations, samples per-particle survival, clears every field of a
removed fixed slot, and supports persistent per-box RNG state.

## In Scope

- **Shipped in P1 / #1401:** Consolidate neutral fp64 particle-transport
  helpers in `particula.gpu.properties`, migrate all consumers, and remove
  legacy `particula.gpu.dynamics` definitions and re-exports. The shipped
  property primitives include defined slip zero/invalid behavior plus
  device-only Debye and rectangular `x_coth_x` geometry factors.
- Validate or add the remaining Warp device primitives needed for coefficient
  assembly, using the P1 property import surface.
- Immutable host configuration for exactly `"spherical"` and `"rectangular"`
  geometry, with SI-unit chamber parameters and wall eddy diffusivity.
- Scalar, same-device per-box, or explicit `WarpEnvironmentData` temperature and
  pressure following existing direct-kernel normalization conventions.
- Deterministic fp64 coefficient evaluation for active particle-resolved slots.
- Stochastic survival with probability `exp(-k * time_step)` and caller-owned
  `(n_boxes,)` `wp.uint32` RNG state that is not implicitly reseeded.
- In-place removal that clears all species masses, concentration, and charge for
  every lost slot while preserving array shapes, devices, dtypes, identities,
  density, volume, and surviving slots.
- Validation-before-allocation/RNG/mutation, CPU coefficient parity, statistical
  survival validation, Warp CPU baseline, and optional CUDA evidence.

## Out of Scope

- Wall-loss coefficient assembly, a direct wall-loss API/configuration,
  removal kernels, and RNG lifecycle in P1; these remain later E6-F3 phases.
- Charged, image-charge, wall-potential, or electric-field physics (E6-F4).
- Discrete or continuous-PDF GPU wall loss, gas wall loss, multi-box transport,
  or changes to CPU strategy behavior.
- High-level GPU `Runnable`, backend selection, scheduler/resident-loop APIs, or
  hidden CPU/GPU transfer and fallback (Epic G).
- Dynamic particle-array resizing, slot compaction or activation, nucleation,
  resampling, graph capture, differentiability, and performance claims.
- Exact CPU/NumPy and Warp RNG-stream matching; only distributional parity is
  required for stochastic outcomes.
