# Scope

Deliver ST1956 turbulent-shear-only execution through E5-F1's shared direct GPU
coagulation contract, with explicit per-box dissipation and fluid-density inputs
and independently checked pair physics, majorant, validation, and conservation.

## In Scope

- A focused fp64 Warp helper for
  `sqrt(pi * dissipation / (120 * kinematic_viscosity)) * (2r_i + 2r_j)^3`.
- Dynamic viscosity from each box's temperature and kinematic viscosity as
  `dynamic_viscosity / fluid_density`.
- Required dissipation and fluid-density inputs accepted as floating scalars or
  active-device Warp arrays shaped `(n_boxes,)`, normalized explicitly.
- Positive-finite shape, dtype, and device validation before allocations,
  kernel launches, particle mutation, output writes, or RNG advancement.
- A safe turbulent-shear majorant and turbulent-shear-only registration in the
  E5 capability matrix and one-pass bounded sampler.
- Warp CPU formula, validation, multi-box, stochastic, conservation, buffer,
  and persistent-RNG evidence; CUDA coverage when available.
- Developer documentation that states units and the bounded ST1956 support
  claim.

## Out of Scope

- Any DNS turbulence implementation or claim, including AO2008/DNS kernels,
  Reynolds-number corrections, clustering, preferential concentration, or
  inertial-particle extensions.
- Brownian-plus-turbulent-shear or larger additive combinations before E5-F6.
- Inferring fluid density from pressure/temperature, storing it in
  `WarpEnvironmentData`, or adding turbulence fields to shared data containers.
- Binned/continuous-PDF execution, high-level strategies or runnables, CPU
  fallback, hidden transfers/synchronization, graph-capture guarantees,
  adaptive stepping, performance redesign, or exact CPU/Warp RNG replay.
