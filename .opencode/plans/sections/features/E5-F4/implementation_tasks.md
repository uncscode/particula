# Implementation Tasks

## GPU Physics and Orchestration

- [x] Add an effective-density helper to
  `particula/gpu/dynamics/coagulation_funcs.py` using total mass divided by
  composition volume; define guards for non-positive inputs.
- [x] Add a Stokes settling helper using the existing fp64 viscosity, mean-free-
  path, Knudsen, and Cunningham-slip helpers plus `STANDARD_GRAVITY`.
- [x] Add a scalar SP2016 pair helper implementing
  `pi * (radius_i + radius_j)^2 * abs(velocity_i - velocity_j)` with no public
  collision-efficiency argument.
- [ ] Extend E5-F1's particle-property preparation to calculate radius,
  effective density, and settling velocity only when sedimentation is enabled.
- [ ] Implement a safe active-pair sedimentation majorant as the exhaustive
  maximum over `i < j`, including zero-rate and non-finite device guards.
- [ ] Add sedimentation to E5-F1's shared mechanism mask, pair-rate dispatch,
  total-majorant scheduling, and one-pass acceptance path.
- [ ] Register only sedimentation-only execution in the capability matrix;
  leave additive combinations reserved for E5-F6.
- [ ] Preserve collision-pair/count buffer identity, persistent RNG semantics,
  inactive slots, existing apply behavior, and the return tuple.
- [ ] Ensure unsupported configuration or invalid density/environment/buffer
  requests fail before allocation, RNG initialization/advancement, output
  mutation, or particle mutation.

## Tooling / Tests

- [x] Add probe-kernel tests for effective density, settling velocity, and
  SP2016 pair rate in
  `particula/gpu/dynamics/tests/coagulation_funcs_test.py`.
- [x] Build expected values with independent NumPy equations; do not call the
  new Warp helper to produce expected results.
- [ ] Add deterministic one-box and multi-box/multi-species parity fixtures,
  including equal settling velocities and composition-dependent densities.
- [ ] Add bounded repeated-run stochastic checks rather than exact CPU/Warp pair
  replay.
- [ ] Add species-mass conservation, donor clearing, inactive-slot, zero/one/
  two-active, caller-buffer, persistent-RNG, and scheduled-trial-cap regressions.
- [ ] Snapshot all caller-owned state around invalid non-unit-efficiency,
  unsupported combination/distribution, shape, dtype, device, and domain cases.
- [ ] Run focused Warp CPU tests when Warp is installed and optional CUDA tests
  with clean skips; retain the repository coverage threshold.

## Documentation

- [ ] Update API docstrings and feature/roadmap documentation with the exact
  equation, efficiency-1 rule, direct import, supported inputs/devices, and
  exclusions.
- [ ] Mark E5-F4 phases and parent E5 dependencies accurately as work ships.
