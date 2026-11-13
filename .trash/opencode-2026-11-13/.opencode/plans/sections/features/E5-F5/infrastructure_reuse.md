# Infrastructure Reuse

- `get_turbulent_shear_kernel_st1956()` and its system-state wrapper in
  `particula/dynamics/coagulation/turbulent_shear_kernel.py:30-138` define the
  independent CPU formula, units, and temperature/fluid-density relationship.
- `get_kinematic_viscosity_via_system_state()` in
  `particula/gas/properties/kinematic_viscosity.py:67-117` defines
  `nu = mu(T) / rho_fluid`; use it as a test oracle, not a runtime dependency.
- `dynamic_viscosity_wp()` in
  `particula/gpu/properties/gas_properties.py:10-36` already supplies the fp64
  Sutherland calculation required before the ST1956 term.
- Radius derivation and active-slot preparation in
  `particula/gpu/kernels/coagulation.py:251-294` reuse species mass/density and
  avoid a new particle sidecar.
- E5-F1's mechanism mask, capability matrix, term dispatch, summed-majorant
  contract, and one-pass active-pair sampler are the required integration seam;
  extend them rather than adding a second selector or public step.
- `_ensure_environment_arrays()` integration at
  `particula/gpu/kernels/coagulation.py:894-907`, `_ensure_volume_array()` at
  lines 745-789, and `_validate_device_match()` at lines 497-510 establish the
  scalar/Warp-array normalization and fail-before-launch pattern to follow for
  dissipation and fluid density.
- Existing output-buffer and persistent-RNG checks at
  `particula/gpu/kernels/coagulation.py:919-978` preserve caller ownership.
- `particula/gpu/kernels/tests/coagulation_test.py` provides device fixtures,
  explicit fp64 data, Warp-absent collection behavior, conservation checks,
  buffer identity checks, and stochastic assertion patterns.
- E5-F4's exhaustive active-pair majorant pattern is the sibling precedent.
  ST1956 also permits the tighter monotone bound at the two largest active
  radii per box; whichever implementation is chosen must be proved by
  all-pairs regression tests before use.
