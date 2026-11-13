# Infrastructure Reuse

- `condensation_step_gpu()` and `condensation_mass_transfer_kernel()` in
  `particula/gpu/kernels/condensation.py:58-178,387-570` are the supported
  boundary and correction insertion point.
- CPU `get_thermal_resistance_factor()` and
  `get_mass_transfer_rate_latent_heat()` in
  `particula/dynamics/condensation/mass_transfer.py:183-331` are authoritative.
- `get_latent_heat_energy_released()` in that module at lines 334-397 defines
  signed `Q = Δm L` semantics.
- `get_thermal_conductivity()` in
  `particula/gas/properties/thermal_conductivity.py:14-45` defines
  `k(T) = 1e-3 * (4.39 + 0.071 T)`.
- `apply_mass_transfer_kernel()` at `condensation.py:217-238` is the current
  update path; transfer must be bounded before application and diagnostics.
- Reuse E4-F3's fixed-substep and scratch pattern, prototyped in
  `particula/gpu/kernels/tests/_condensation_test_support.py:338-370,804-848`.
- Keep `WarpParticleData`, `WarpGasData`, and `WarpEnvironmentData` in
  `particula/gpu/warp_types.py:24-169` unchanged; use sidecar arrays.
- Follow lazy kernel exports, fp64 arrays, device validation, and
  validate-before-allocation/mutation conventions.
