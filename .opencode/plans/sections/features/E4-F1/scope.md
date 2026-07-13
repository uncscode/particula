# Scope

Issue #1281 shipped only the E4-F1-P1 validation boundary. It supplies a
caller-owned fixed-shape sidecar and validates it before condensation continues
through its existing path.

## In Scope

- Numeric, fixed-shape, species-indexed model modes and parameters.
- Validation of mode, parameter bounds, dtype, shape, species count/order, and
  Warp device before launch or output mutation.
- `ThermodynamicsConfig` in `particula/gpu/kernels/thermodynamics.py`, with
  `modes` `(n_species,)` `wp.int32`, `parameters` `(n_species, 4)`
  `wp.float64`, and ordered `molar_mass_reference` `(n_species,)` `wp.float64`.
- Validation before optional species defaults, caller `mass_transfer` access,
  scratch allocation, formula work, or `wp.launch`.
- A required keyword-only `thermodynamics` argument to
  `condensation_step_gpu()`; omission raises `ValueError`.
- Migration of executable GPU benchmark and quick-start calls plus focused
  validator and condensation-boundary regression coverage.

## Out of Scope

- Activity and surface-tension physics (E4-F2).
- Four-substep production orchestration and scratch management (E4-F3).
- Latent heat (E4-F4), gas coupling/conservation (E4-F5), broad readiness
  evidence (E4-F6), and final user examples/support matrix (E4-F7).
- Porting CPU vapor-pressure strategies other than constant and Buck.
- Moving vapor pressure into CPU `GasData` or storing Python strategy objects,
  strings, or species names in Warp data.
- Calculating vapor pressure, refreshing `WarpGasData.vapor_pressure`, launching
  a formula kernel, or modifying particle/gas container schemas.
- User-facing thermodynamics documentation or migration-guide updates.
