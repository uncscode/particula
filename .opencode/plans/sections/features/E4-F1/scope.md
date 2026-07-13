# Scope

Issues #1281 and #1282 shipped E4-F1-P1/P2: a caller-owned fixed-shape sidecar
and a standalone, validated on-device vapor-pressure refresh primitive.

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
- `refresh_vapor_pressure_gpu` in
  `particula/gpu/kernels/thermodynamics.py`, accepting Warp `float64` gas
  buffers and `(n_boxes,)` temperature on the gas device.
- One `(n_boxes, n_species)` launch that overwrites the pressure matrix using
  constant Pa values or canonical Buck water/ice equations.
- Co-located parity, API-surface, overwrite, and failure-before-mutation tests.

## Out of Scope

- Activity and surface-tension physics (E4-F2).
- Four-substep production orchestration and scratch management (E4-F3).
- Latent heat (E4-F4), gas coupling/conservation (E4-F5), broad readiness
  evidence (E4-F6), and final user examples/support matrix (E4-F7).
- Porting CPU vapor-pressure strategies other than constant and Buck.
- Moving vapor pressure into CPU `GasData` or storing Python strategy objects,
  strings, or species names in Warp data.
- Pre-step integration in `condensation_step_gpu()`: after successful input and
  sidecar validation, it refreshes caller-owned `gas.vapor_pressure` exactly
  once from normalized current per-box device temperature before environment
  preparation and mass transfer. Direct `wp.float32` temperature is copied into
  a device-local `wp.float64` buffer for the refresh boundary.
- User-facing thermodynamics documentation or migration-guide updates.
