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
- [x] Add private cleared settling-velocity scratch and calculate guarded
  properties only for the exact private sedimentation-only mask.
- [x] Implement a safe compact active-pair sedimentation majorant as the
  exhaustive maximum over `i < j`, including zero-rate and non-finite guards.
- [x] Add internal sedimentation pair-rate/majorant dispatch to the shared
  bounded scheduler and one-pass acceptance/RNG path.
- [x] Register only the exact public sedimentation-only capability; reject mixed
  or unsupported sedimentation configurations before particle runtime access.
- [x] Preserve collision-pair/count buffer identity, persistent RNG semantics,
  inactive slots, and existing apply behavior in the private exact-mask path.
- [x] Ensure unsupported configuration or invalid density/environment/buffer
  requests fail before allocation, RNG initialization/advancement, output
  mutation, or particle mutation.

## Tooling / Tests

- [x] Add probe-kernel tests for effective density, settling velocity, and
  SP2016 pair rate in
  `particula/gpu/dynamics/tests/coagulation_funcs_test.py`.
- [x] Build expected values with independent NumPy equations; do not call the
  new Warp helper to produce expected results.
- [x] Add co-located private-dispatch coverage for the exhaustive majorant,
  bounded scheduling/RNG behavior, scratch clearing, and mixed-mask no-op.
- [x] Add end-to-end multi-box conservation and broader state-safety evidence
  for the public capability.
- [x] Snapshot all caller-owned state around invalid non-unit-efficiency,
  unsupported combination/distribution, shape, dtype, device, and domain cases.
- [x] Run focused Warp CPU tests when Warp is installed and optional CUDA tests
  with clean skips; retain the repository coverage threshold.

## Documentation

- [x] Publish the canonical direct-kernel contract in
  `docs/Features/data-containers-and-gpu-foundations.md`, including ownership,
  device, and unsupported-boundary wording.
- [x] Mark E5-F4 delivery evidence accurately as work ships.
