# Infrastructure Reuse

- `get_particle_resolved_wall_loss_step()` in
  `particula/dynamics/wall_loss/wall_loss_strategies.py:45-123` defines the CPU
  survival probability, active-slot filtering, and stochastic removal oracle.
- `WallLossStrategy.step()` in
  `particula/dynamics/wall_loss/wall_loss_strategies.py:265-341` defines CPU
  particle-resolved mutation semantics; strengthen the GPU invariant by also
  clearing charge when a slot is removed.
- `SphericalWallLossStrategy` and `RectangularWallLossStrategy` in
  `particula/dynamics/wall_loss/wall_loss_strategies.py:349-488` and
  `:1029-1185` provide geometry validation and coefficient reference behavior.
- `get_spherical_wall_loss_coefficient*()` and
  `get_rectangle_wall_loss_coefficient*()` in
  `particula/dynamics/properties/wall_loss_coefficient.py:35-306` are the
  authoritative Crump-Seinfeld equations and system-state oracle.
- Existing Warp physics functions in `particula/gpu/properties/` and
  `particula/gpu/dynamics/condensation_funcs.py` provide viscosity, Knudsen,
  Cunningham slip, diffusion, and settling building blocks; P1 must inventory
  these and add only genuinely missing neutral primitives such as Debye or a
   numerically safe coth helper.
- Shipped P2 `particula/gpu/dynamics/wall_loss_funcs.py` composes the P1
  property import surface with `settling_velocity_stokes_from_transport_wp`;
  it does not duplicate transport formulas or use the convenience settling
  wrapper that recomputes gas-state transport.
- `WarpParticleData` and `WarpEnvironmentData` in
  `particula/gpu/warp_types.py:24-78` and `:164-184` define fixed fp64 shapes,
  per-box environment state, and caller ownership.
- `initialize_coagulation_rng_states()` and `coagulation_step_gpu()` in
  `particula/gpu/kernels/coagulation.py:1932-1972` and `:2104-2364` provide the
  persistent `(n_boxes,)` `wp.uint32` RNG lifecycle, scalar/per-box environment
  normalization, and validation-before-RNG pattern to mirror.
- `apply_coagulation_kernel()` in
  `particula/gpu/kernels/coagulation.py:1345-1414` is prior art for clearing all
  donor masses, concentration, and charge without resizing fixed arrays.
- `particula/gpu/kernels/__init__.py:18-35` provides lazy public entry-point
  export conventions; expose only the direct step, not configuration helpers.
- `particula/gpu/tests/cuda_availability.py:14-37` supplies mandatory Warp CPU
  and optional cleanly-skipping CUDA parameterization.
- `particula/gpu/kernels/tests/coagulation_validation_test.py:722-771` provides
  repeated-call persistent-RNG tests; CPU wall-loss coefficient and strategy
  tests under `particula/dynamics/**/wall_loss*_test.py` provide fixtures and
  independent expected values.
