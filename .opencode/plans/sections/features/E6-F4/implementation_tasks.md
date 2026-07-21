# Implementation Tasks

## GPU Physics

- [ ] Inventory E6-F3 symbols and extend only its concrete wall-loss modules.
- [ ] Add fp64 image-charge factor helper matching CPU diagonal extraction,
  absolute value, clipping, exponentiation, and zero-charge identity.
- [ ] Add geometry-scale and resolved-field helpers matching scalar/vector and
  wall-potential semantics.
- [ ] Add signed electric mobility/drift helper with radius and scale guards.
- [ ] Compose `neutral * electrostatic_factor + drift` with CPU-equivalent NaN
  sanitization and finite nonnegative clipping.

## Kernel Contract

- [ ] Extend the E6-F3 immutable configuration in
  `particula/gpu/kernels/wall_loss.py` without adding a second step API.
- [ ] Validate charged capability, field form, finite charge, particle schema,
  environment, time, and RNG sidecar before allocation or mutation.
- [ ] Preserve exact E6-F3 neutral computation for each zero-charge slot.
- [ ] Read charge without copying caller arrays or changing device ownership.
- [ ] Reuse active-slot, stochastic survival, removal-clearing, and persistent-
  RNG kernels; ensure removed slots clear every species mass, concentration,
  and charge while survivor fields retain identity and value.
- [ ] Preserve exact zero-time and all-inactive no-op behavior, including no RNG
  advancement.

## Tooling / Tests

- [ ] Add primitive parity cases to
  `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`.
- [ ] Add configuration, preflight, mixed-charge, identity, clearing, and RNG
  lifecycle cases to `particula/gpu/kernels/tests/wall_loss_test.py`.
- [ ] Extend `particula/gpu/kernels/tests/wall_loss_parity_test.py` with a
  deterministic CPU/Warp geometry/charge/field matrix.
- [ ] Compare survival counts with predeclared binomial confidence or sigma
  bounds over enough samples/seeds; never compare NumPy and Warp draw order.
- [ ] Require Warp CPU and use clean optional CUDA skips.
- [ ] Keep CPU wall-loss, E6-F3 neutral, coagulation RNG, and fixed-slot
  regression suites green without lowering coverage or tolerances.

## Documentation

- [ ] Document direct charged configuration, SI units, image-charge at zero
  potential, field semantics, and neutral fallback.
- [ ] Document caller-owned charge/RNG state and failure-before-mutation limits.
- [ ] Publish focused validation commands and explicit supported/deferred scope.
