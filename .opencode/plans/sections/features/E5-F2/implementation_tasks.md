# Implementation Tasks

## GPU Physics

- [ ] Implement `charged_hard_sphere` as the sole initial GPU charged model.
  Keep Dyachkov, Gatti, Gopalakrishnan, and Chahl variants unavailable until a
  separately approved parity contract exists for each one.
- [x] Add scalar fp64 Coulomb potential and stable enhancement-limit helpers in
  `particula/gpu/dynamics/coagulation_funcs.py`.
- [x] Add scalar reduced mass and reduced friction helpers with explicit
  zero/positive-domain guards in the same module.
- [x] Port the scalar diffusive Knudsen calculation from the
  published CPU formula chain without calling NumPy code at runtime.
- [ ] Add the scalar `charged_hard_sphere` pair helper, returning a finite
  non-negative rate in cubic meters per second; do not add generic or deferred
  charged-model dispatch.

## Kernel and Validation

- [ ] Extend `_validate_particle_arrays` in
  `particula/gpu/kernels/coagulation.py` to require charge shape
  `(n_boxes, n_particles)` and `wp.float64` storage.
- [ ] Extend `_validate_device_arrays` to require charge on the masses device.
- [ ] Add active-device finite-value preflight for charge before volume setup,
  RNG initialization, work allocation, or coagulation launches.
- [ ] Add `particles.charge` to `apply_coagulation_kernel`; sum recipient and
  donor charge and zero donor charge before returning.
- [ ] Keep the public `coagulation_step_gpu` return tuple, collision buffers,
  persistent RNG behavior, and Brownian default unchanged.

## Tooling / Tests

- [x] Add scalar helper probe kernels and independent CPU/Warp parity tables to
  `particula/gpu/dynamics/tests/coagulation_funcs_test.py`.
- [x] Test neutral, same-sign, opposite-sign, clipped repulsion, equal and
  mixed scales, zero/negative safe branches, and the extreme kinetic threshold
  for the P1 primitives.
- [ ] Add malformed shape/dtype/device/non-finite charge cases to
  `particula/gpu/kernels/tests/coagulation_test.py` with complete before/after
  state and persistent-RNG snapshots.
- [ ] Add deterministic direct `apply_coagulation_kernel` tests for donor clear,
  recipient sum, no-collision behavior, multiple species, and multiple boxes.
- [ ] Extend step-level conservation tests to assert species mass and charge
  separately per box on Warp CPU and optional CUDA.
- [ ] Run focused fast tests, Ruff, and mypy without lowering the configured
  coverage threshold.
