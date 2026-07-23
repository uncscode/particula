# Implementation Tasks

## GPU Physics and Kernel API

- [ ] Inventory existing Warp viscosity, mean-free-path, slip, diffusion, radius,
  effective-density, and settling functions; document exact CPU-reference gaps.
- [x] Add the required internal neutral fp64 coefficient helpers in
  `particula/gpu/dynamics/wall_loss_funcs.py`, reusing P1 zero-limit-safe
  geometry primitives.
- [x] Implement spherical and rectangular coefficient functions matching
  `particula/dynamics/properties/wall_loss_coefficient.py` term by term.
- [x] Define immutable neutral geometry configuration in
  `particula/gpu/kernels/wall_loss.py` without charged fields or container state.
- [x] Add structural, shape, dtype, device, and finite-domain validators that
  complete before output allocation, RNG work, or mutation.
- [x] Validate direct scalar/per-box environment inputs or explicit
  `WarpEnvironmentData` without hidden host transfer or fallback.
- [ ] Implement active-slot coefficient and stochastic survival kernels using
  `exp(-k * dt)` and one per-active-slot decision.
- [ ] Implement a removal kernel that zeros every species mass, concentration,
  and charge for lost slots and never changes shape or survivor fields.
- [x] Add P3 RNG metadata validation without initialization or advancement;
  persistent lifecycle remains deferred.
- [x] Add `wall_loss_step_gpu` to the lazy mapping in
  `particula/gpu/kernels/__init__.py`; do not top-level export configuration or
  introduce a high-level GPU runnable.
- [x] Replace P4's local seed/slot draw path with a sequential per-box removal
  mask that advances caller-owned RNG state only for eligible slots.
- [x] Resolve omitted private state and supplied-state explicit reset only after
  successful positive-time preflight; preserve supplied sidecars for zero time
  and rejected calls.

## Tooling and Tests

- [ ] Add device-primitive tests in co-located `particula/gpu/**/tests/` modules
  against independent NumPy/CPU property functions.
- [x] Add `particula/gpu/dynamics/tests/wall_loss_funcs_test.py` for guarded
  deterministic spherical/rectangular CPU/Warp parity and smoke launches over
  scalar diffusion/gravity and vector nanometer-to-micrometer states.
- [x] Add `particula/gpu/kernels/tests/wall_loss_test.py` for configuration,
  preflight ordering, and unchanged particle/sidecar state on valid and invalid
  P3 calls. Removal, inactive-gap execution, and RNG lifecycle tests remain
  P4-P5 work.
- [x] Add `particula/gpu/kernels/tests/wall_loss_parity_test.py` for independent
  one-/multi-box coefficient/eligibility diagnostics, 100-seed fresh and
  persistent-RNG survival statistics, exact no-ops, and lazy-import smoke
  coverage (#1406).
- [x] Snapshot particles and supplied RNG before invalid P3 calls and assert
  value equality plus identity retention after rejection.
- [x] Require Warp CPU using `warp_devices()` and run CUDA cases only when
  available, with no CUDA-only acceptance criterion (#1406).
- [ ] Keep configured coverage at or above 80%; do not weaken thresholds or
  stochastic assertions to accommodate implementation defects.

## Documentation

- [ ] Document units, formulas, configuration exclusivity, active-slot
  predicate, RNG ownership/reset semantics, asynchronous mutation boundary,
  and exact fields cleared on removal.
- [ ] Publish focused Warp CPU commands and supported/deferred tables; link
  E6-F3 to parent E6, downstream E6-F4, and closeout E6-F9.

## P4 Completed Tasks (#1404)

- [x] Implement usable-slot neutral coefficient/mask and separate clearing
  kernels in `particula/gpu/kernels/wall_loss.py` after frozen P3 preflight.
- [x] Use deterministic local seed/slot survival draws and preserve optional
  `rng_states` without initialization or advancement.
- [x] Extend `particula/gpu/kernels/tests/wall_loss_test.py` for both geometries,
  masks, sparse slots, zero time, controlled removal/survival, aggregate
  stochastic behavior, and pre-launch atomicity.
- [x] Update bounded P4 contract documentation in the index and architecture
  guides without broadening exports or API boundaries.

## P5 Completed Tasks (#1405)

- [x] Add private per-box Warp RNG initialization and persistent supplied-sidecar
  lifecycle in `particula/gpu/kernels/wall_loss.py`.
- [x] Extend `particula/gpu/kernels/tests/wall_loss_test.py` with omitted-state,
  initialize-once/reuse, explicit-reset, per-box, eligible-only, all-ineligible,
  zero-time, rejection, and benchmark-smoke coverage.
- [x] Update user and contract documentation for sidecar ownership, explicit
  reset, sequential eligible-slot advancement, and bounded performance scope.

## P6 Completed Tasks (#1406)

- [x] Add test-only guarded Warp coefficient diagnostics against independent CPU
  wall-loss equations for both neutral geometries and recorded tolerances.
- [x] Add 100-seed fresh-state and persistent-sidecar statistical survival
  checks with fixed 3-sigma bounds, plus exact zero-time/all-inactive no-ops.
- [x] Smoke-test the lazy `wall_loss_step_gpu` export and concrete-only
  `NeutralWallLossConfig` boundary without changing production modules.
