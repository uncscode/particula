# Implementation Tasks

## GPU Physics and Kernel API

- [ ] Inventory existing Warp viscosity, mean-free-path, slip, diffusion, radius,
  effective-density, and settling functions; document exact CPU-reference gaps.
- [x] Add the required internal neutral fp64 coefficient helpers in
  `particula/gpu/dynamics/wall_loss_funcs.py`, reusing P1 zero-limit-safe
  geometry primitives.
- [x] Implement spherical and rectangular coefficient functions matching
  `particula/dynamics/properties/wall_loss_coefficient.py` term by term.
- [ ] Define immutable neutral geometry configuration in
  `particula/gpu/kernels/wall_loss.py` without charged fields or container state.
- [ ] Add structural, shape, dtype, device, finite-domain, and active-slot
  validators that complete before output allocation, RNG work, or mutation.
- [ ] Normalize direct scalar/per-box environment inputs or explicit
  `WarpEnvironmentData` without hidden host transfer or fallback.
- [ ] Implement active-slot coefficient and stochastic survival kernels using
  `exp(-k * dt)` and one per-active-slot decision.
- [ ] Implement a removal kernel that zeros every species mass, concentration,
  and charge for lost slots and never changes shape or survivor fields.
- [ ] Add persistent RNG validation/initialization modeled on direct coagulation;
  preserve supplied sidecar identity and avoid implicit repeated-call reseeding.
- [ ] Add `wall_loss_step_gpu` to the lazy mapping in
  `particula/gpu/kernels/__init__.py`; do not top-level export configuration or
  introduce a high-level GPU runnable.

## Tooling and Tests

- [ ] Add device-primitive tests in co-located `particula/gpu/**/tests/` modules
  against independent NumPy/CPU property functions.
- [x] Add `particula/gpu/dynamics/tests/wall_loss_funcs_test.py` for guarded
  deterministic spherical/rectangular CPU/Warp parity and smoke launches over
  scalar diffusion/gravity and vector nanometer-to-micrometer states.
- [ ] Add `particula/gpu/kernels/tests/wall_loss_test.py` for configuration,
  preflight ordering, inactive gaps, exact no-ops, complete slot clearing,
  unchanged survivor/container state, and RNG lifecycle.
- [ ] Add `particula/gpu/kernels/tests/wall_loss_parity_test.py` for one/multi-box
  deterministic coefficients and statistically bounded survival frequencies.
- [ ] Snapshot particles and supplied RNG before every invalid-call case and
  assert byte/array equality plus identity retention after rejection.
- [ ] Require Warp CPU using `warp_devices()` and run CUDA cases only when
  available, with stable skip reasons and no CUDA-only acceptance criterion.
- [ ] Keep configured coverage at or above 80%; do not weaken thresholds or
  stochastic assertions to accommodate implementation defects.

## Documentation

- [ ] Document units, formulas, configuration exclusivity, active-slot
  predicate, RNG ownership/reset semantics, asynchronous mutation boundary,
  and exact fields cleared on removal.
- [ ] Publish focused Warp CPU commands and supported/deferred tables; link
  E6-F3 to parent E6, downstream E6-F4, and closeout E6-F9.
