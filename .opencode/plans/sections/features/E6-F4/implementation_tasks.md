# Implementation Tasks

## GPU Physics

- [x] Inventory E6-F3 symbols and extend only its concrete wall-loss modules.
- [x] Add private fp64 Coulomb self-potential-ratio and image-charge helpers in
  `particula/gpu/dynamics/wall_loss_funcs.py`, matching the CPU self-pair
  ratio floor, absolute value, exponent clipping, and zero-charge identity.
- [x] Add private geometry-scale and resolved-field helpers matching
  scalar/vector and wall-potential semantics.
- [x] Add private signed electric mobility/drift helper with radius and scale
  guards.
- [x] Compose `neutral * electrostatic_factor + drift` with CPU-equivalent NaN
  sanitization and finite nonnegative clipping.

## Kernel Contract

- [x] Extend the E6-F3 immutable configuration in
  `particula/gpu/kernels/wall_loss.py` without adding a second step API.
- [x] Validate neutral/charged capability, potential/field form and ownership,
  particle schema, field-before-charge ordering, environment, time, and RNG
  sidecar before allocation or mutation.
- [x] Preserve the exact E6-F3 neutral computation and RNG path while executing
  charged physics only for nonzero-charge slots.
- [x] Read charge without copying caller arrays or changing device ownership.
- [x] Reuse active-slot, stochastic survival, removal-clearing, and persistent-
  RNG kernels; ensure removed slots clear every species mass, concentration,
  and charge while survivor fields retain identity and value.
- [x] Preserve exact zero-time and all-inactive no-op behavior, including no RNG
  advancement.

## Tooling / Tests

- [x] Add independent NumPy/Warp primitive parity and clipping cases in
  `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`, including direct
  ratio-floor coverage, zero charge, signed equal-magnitude charge, and both
  clipping domains.
- [x] Add P1 configuration, preflight, ordering, ownership, atomicity, and
  zero-charge neutral-equivalence cases to
   `particula/gpu/kernels/tests/wall_loss_test.py`.
- [x] Add independent P3 NumPy/Warp field-resolution, signed-drift, guard, and
  defensive composition tests in `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`.
- [x] Extend `particula/gpu/kernels/tests/wall_loss_parity_test.py` with an
  independent deterministic charged CPU/Warp geometry/charge/field matrix,
  including snapshots proving particle and caller-owned field non-mutation.
- [x] Complete the frozen eight-stratum charged survival validation with
  predeclared inclusive exact-binomial bounds (4,096 observations per stratum);
  never compare NumPy and Warp draw order.
- [x] Require Warp CPU and use clean optional CUDA skips.
- [x] Keep CPU wall-loss, E6-F3 neutral, coagulation RNG, and fixed-slot
  regression suites green without lowering coverage or tolerances.

## Documentation

- [x] Document direct charged configuration, SI units, image-charge at zero
  potential, field semantics, and neutral fallback.
- [x] Document caller-owned charge/RNG state and failure-before-mutation limits.
- [x] Publish focused validation commands and explicit supported/deferred scope.
