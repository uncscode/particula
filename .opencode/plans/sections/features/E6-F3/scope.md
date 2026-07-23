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
- **Shipped in P2 / #1402:** Add concrete internal fp64 Warp spherical and
  rectangular neutral Crump-Seinfeld coefficient helpers in
  `particula/gpu/dynamics/wall_loss_funcs.py`, with guarded CPU/Warp parity and
   smoke coverage in `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`.
- **Shipped in P3 / #1403:** Add frozen concrete-module
  `NeutralWallLossConfig`, write-free `wall_loss_step_gpu` preflight in
  `particula/gpu/kernels/wall_loss.py`, its lazy kernel-package export, and
  Warp-guarded atomicity contract coverage in
  `particula/gpu/kernels/tests/wall_loss_test.py`. P3 accepts only neutral
  particle-resolved spherical or rectangular configuration and validates
  particle, environment, time, and optional RNG-sidecar metadata without
  mutable-runtime work.
- Validate or add the remaining Warp device primitives needed for coefficient
  assembly, using the P1 property import surface.
- Deterministic fp64 coefficient evaluation for active particle-resolved slots.
- Stochastic survival with probability `exp(-k * time_step)` and caller-owned
  `(n_boxes,)` `wp.uint32` RNG state that is not implicitly reseeded.
- In-place removal that clears all species masses, concentration, and charge for
  every lost slot while preserving array shapes, devices, dtypes, identities,
  density, volume, and surviving slots.
- Validation-before-allocation/RNG/mutation, CPU coefficient parity, statistical
  survival validation, Warp CPU baseline, and optional CUDA evidence.
- **Shipped in P6 / #1406:** Add the test-only
  `particula/gpu/kernels/tests/wall_loss_parity_test.py` diagnostic matrix.
  It independently compares complete-slot coefficient/eligibility results with
  CPU system-state equations; tests fresh-seed and persistent-sidecar aggregate
  survival over 100 seeds; retains exact no-op checks; and smoke-tests the lazy
  export boundary. Production code, physics, and exports are unchanged.

## Out of Scope

- Charged, image-charge, wall-potential, or electric-field physics (E6-F4).
- Discrete or continuous-PDF GPU wall loss, gas wall loss, multi-box transport,
  or changes to CPU strategy behavior.
- High-level GPU `Runnable`, backend selection, scheduler/resident-loop APIs, or
  hidden CPU/GPU transfer and fallback (Epic G).
- Dynamic particle-array resizing, slot compaction or activation, nucleation,
  resampling, graph capture, differentiability, and performance claims.
- Exact CPU/NumPy and Warp RNG-stream matching; only distributional parity is
  required for stochastic outcomes.

## P4-P5 Delivered Scope (#1404, #1405)

After frozen P3 preflight, positive-time neutral particle-resolved calls
normalize environment inputs, calculate spherical or rectangular coefficients
for usable active slots, and clear all species masses, concentration, and charge
for lost slots. P5 supplies the persistent RNG contract: omitted state is private
and initialized per successful call; supplied `(n_boxes,)` `wp.uint32` state is
caller-owned, advances sequentially by box for eligible slots only, and resets
only with `initialize_rng=True`. Zero time and rejected preflight leave supplied
state unchanged. The serial per-box RNG loop is bounded correctness scope, not a
performance claim.

## P6 Delivered Scope (#1406)

P6 is validation-only. The parity suite covers spherical/rectangular geometry,
one-/multi-box layouts, per-box state, nanometer/micrometer scales, and
inactive/unusable slots, with Warp CPU baseline and optional CUDA parameter rows.
It deliberately makes distributional—not stream-replay—claims for stochastic
survival and does not add a production diagnostic, API, transfer, fallback, or
public export.
