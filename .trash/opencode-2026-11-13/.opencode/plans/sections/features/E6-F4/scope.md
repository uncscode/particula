# Scope

Extend E6-F3's low-level particle-resolved Warp wall-loss operation in staged
phases. P1-P4 are implemented: configuration/preflight semantics and private
fp64 charged-coefficient primitives are frozen, and charged selection now
executes in the existing fixed-shape direct step without changing neutral mode.

## In Scope

- **Shipped P1:** `NeutralWallLossConfig` has appended `mode`,
  `wall_potential`, and `wall_electric_field` fields, preserving legacy
  positional construction and remaining concrete-module-only.
- **Shipped P1:** neutral/charged mode validation; finite signed scalar
  potential; charged spherical scalar field; and charged rectangular,
  caller-owned same-device `wp.float64` `(3,)` field validation.
- **Shipped P1:** staged rectangular-field validation and finite device scan
  occur after particle schema/device discovery but before particle value scans,
  environment/RNG work, output allocation, update-kernel launch, or caller
  mutation. Private validation scans/status scratch may allocate. Rejections
  preserve supplied particle, field, and RNG state.
- **Shipped P4 (#1412):** geometry-specialized charged removal kernels
  dispatch after successful positive-time preflight. Nonzero charged slots use
  the private image-enhancement, field-resolution, signed-drift, and safe
  composition helpers; exact zero-charge slots retain the neutral coefficient
  and RNG path.
- **Shipped P4 (#1412):** neutral mode retains its unchanged kernel and
  launch. Charged spherical launches receive scalar field data only; charged
  rectangular launches read the validated caller-owned `(3,)` vector by
  identity, without copying or mutating it.
- **Shipped P2:** private `@wp.func` helpers implement the fp64 Coulomb
  self-potential ratio and image enhancement with CPU-equivalent clipping and
  exact zero-charge identity. Independent NumPy/Warp parity and clipping tests
   live in `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`.
- **Shipped P3:** private fp64 `@wp.func` helpers resolve geometry scale and
  spherical/rectangular fields, calculate signed mobility drift, and compose
  finite nonnegative charged coefficients with CPU-equivalent sanitization.
  Independent tests cover ordinary, zero, guard-boundary, and defensive
  nonfinite/overflow lanes in `wall_loss_funcs_test.py`.
- Integration with E6-F3's active predicate, fixed-shape removal clearing,
  environment normalization, preflight ordering, and caller-owned RNG lifecycle.
- **Shipped P4 (#1412):** active eligible slots draw at most once; selected
  slots clear all mass lanes, concentration, and charge. Nonpositive composed
  rates consume no draw; saturated charged coefficients use the charged
  survival-draw path, unlike the neutral positive-infinity shortcut.
- **Shipped P5 (#1413):**
  `particula/gpu/kernels/tests/wall_loss_parity_test.py` adds independent CPU
  strategy versus non-mutating Warp charged-coefficient parity for an explicit
  spherical/rectangular matrix. It checks particle and caller-owned rectangular
  field non-mutation, uses spherical `rtol=1.002e-3, atol=1e-20` and rectangular
  `rtol=1e-6, atol=0`, and retains Warp CPU with optional clean CUDA skips.
- **Shipped P5 (#1413):** exact zero-charge charged/neutral diagnostic,
  survivor-state, RNG-sidecar, and rectangular-field ownership equality;
  invalid/no-op non-mutation regressions; and four-radius by two-geometry
  charged survival evidence. Each of the eight strata uses 16 fresh seeded
  fixtures of 256 slots (4,096 observations) and an inclusive equal-tail exact
  binomial interval at per-stratum alpha `1.25e-7`.
- Focused support/deferred documentation and direct-kernel import coverage.

## Out of Scope

- Changes to the CPU charged or neutral wall-loss equations or public strategy.
- Public exports/API, a second direct-step entry point, container schema
  changes, hidden transfer/fallback, or a separate RNG stream.
- Discrete/continuous distributions, a GPU `Runnable`, backend selection,
  scheduling, adaptive stepping, multi-box transport, or process composition.
- Dynamic particle allocation, resizing, compaction, activation, resampling, or
  exhaustion handling; E6-F5/E6-F6 own those capabilities.
- Hidden CPU/GPU transfers, host coefficient fallback, container schema changes,
  hidden RNG ownership, or exact NumPy/Warp random-sequence matching.
- General electrostatic chamber/CFD coupling, alternate charging models,
  graph capture, autodiff, mandatory CUDA, benchmarks, or performance claims.
