# Infrastructure Reuse

- `WarpParticleData.charge` in `particula/gpu/warp_types.py:24-78` already owns
  fp64 charge arrays shaped `(n_boxes, n_particles)`; do not add a sidecar or
  alter conversion schemas.
- `coagulation_step_gpu` validation/allocation/launch ordering in
  `particula/gpu/kernels/coagulation.py:792-1022` is the integration boundary;
  retain fail-before-launch behavior and caller-owned RNG/output contracts.
- `_validate_particle_arrays`, `_validate_device_arrays`, and
  `_validate_device_match` in `particula/gpu/kernels/coagulation.py:468-529`
  are the existing preflight pattern to extend for charge.
- `apply_coagulation_kernel` in
  `particula/gpu/kernels/coagulation.py:430-465` already merges each species'
  mass and clears donor concentration; extend this one mutation site.
- Existing scalar Warp ports in
  `particula/gpu/dynamics/coagulation_funcs.py:10-139` establish the module,
  fp64 type, naming, and unit-test pattern for pair helpers.
- `_system_state_properties` in
  `particula/dynamics/coagulation/charged_dimensional_kernel.py:32-121`
  defines the CPU property pipeline: radius, total mass, friction, Coulomb
  potential, diffusive Knudsen number, and reduced pair quantities.
- `get_coulomb_enhancement_ratio` and stable kinetic/continuum limits in
  `particula/particles/properties/coulomb_enhancement.py:27-186` are independent
  formula references, including the repulsive lower clip at `-200`.
- The approved CPU strategy implementations are cataloged in
  `particula/dynamics/coagulation/charged_kernel_strategy.py:133-305`; only the
  subset frozen by the E5-F1/F2 support decision is ported.
- CPU merge semantics in
  `particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py:790-799`
  copy donor mass/charge, clear donor state, and add both to the recipient.
- Device parametrization and conservation patterns in
  `particula/gpu/kernels/tests/coagulation_test.py:2562-2651` should be extended
  to assert mass and charge separately on Warp CPU and optional CUDA.
