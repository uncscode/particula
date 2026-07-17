# Infrastructure Reuse

- `get_sedimentation_kernel_sp2016()` in
  `particula/dynamics/coagulation/sedimentation_kernel.py` is the
  independent NumPy equation reference. Use its unit-efficiency branch, not its
  unimplemented collision-efficiency callback.
- `get_particle_settling_velocity()` and
  `get_particle_settling_velocity_via_system_state()` in
  `particula/particles/properties/settling_velocity.py` define the approved
  Stokes, gas-property, Knudsen, and slip-correction path.
- `brownian_coagulation_kernel()` in
  `particula/gpu/kernels/coagulation.py` already computes composition
  volume/radius, compacts active indices, schedules bounded trials, selects
  disjoint active pairs, and advances one RNG stream per box. Extend E5-F1's
  factored replacement rather than creating a second selector.
- `coagulation_step_gpu()` in
  `particula/gpu/kernels/coagulation.py` is the concrete low-level
  validation, allocation, launch, and return boundary. Preserve its environment
  input modes and caller-owned collision/RNG buffers.
- `_bound_scheduled_trials()` and active-index swap-pop helpers in
  `particula/gpu/kernels/coagulation.py` provide the bounded sampling and
  disjoint-pair primitives.
- `dynamic_viscosity_wp()` and `molecule_mean_free_path_wp()` in
  `particula/gpu/properties/gas_properties.py`, plus
  `knudsen_number_wp()`, `cunningham_slip_correction_wp()`, and
  `aerodynamic_mobility_wp()` in
  `particula/gpu/properties/particle_properties.py`, provide existing fp64
  device property calculations.
- `particle_radius_from_volume_wp()` in
  `particula/gpu/dynamics/condensation_funcs.py` converts composition volume to
  radius without a host round trip.
- `effective_density_wp()` and `settling_velocity_stokes_wp()` in
  `particula/gpu/dynamics/coagulation_funcs.py` establish the pattern for small
  scalar `@wp.func` pair/property helpers with independent CPU references and
  focused probe-kernel tests.
- `_available_warp_devices()` and `_make_particle_data()` in
  `particula/gpu/kernels/tests/coagulation_test.py` provide device
  parametrization and deterministic multi-species particle fixtures. Follow its
  Warp-optional collection, Warp CPU requirement, optional CUDA, buffer, RNG,
  and conservation patterns.
- E5-F1 supplies canonical mechanism configuration, capability validation,
  additive pair-rate dispatch, and one-pass sampling. E5-F3 demonstrates how a
  mechanism term must extend that shared contract without forking it.
