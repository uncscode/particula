# Implementation Tasks

## GPU Physics and API

- [x] Define stable `wp.int32` constant and Buck mode codes in the concrete
  thermodynamics module.
- [x] Add a frozen, caller-owned fixed-shape configuration with typed,
  species-indexed Warp fields.
- [x] Validate required configuration, supported modes, finite/nonnegative
  values, dtype, shape, species ordering, and active Warp device.
- [x] Implement constant and piecewise canonical Buck Warp calculations from the
  exact CPU references using `float64`.
- [x] Implement a refresh launch that writes every `(box, species)` output slot.
- [x] Add required keyword-only `thermodynamics` to `condensation_step_gpu()`.
- [x] Invoke validation after active-context setup and before defaults,
  mass-transfer access, allocation, or launch.
- [x] Ensure all validation failures occur before `gas.vapor_pressure`, gas
  concentration, or particle mass can mutate.
- [x] Refresh `gas.vapor_pressure` exactly once after all successful step-input
  validation and before environment-property preparation and mass transfer.
- [x] Preserve direct `wp.float32` temperature support with a device-local
  float32-to-float64 copy for the refresh API boundary.

## Tooling / Tests

- [x] Add focused validator unit tests in
  `particula/gpu/kernels/tests/thermodynamics_test.py`.
  or a focused `*_test.py` beside the new module.
- [x] Add mixed-model, multi-box, freezing-boundary, species-order, API, and
  no-mutation refresh tests.
- [x] Add invalid/missing configuration and failure-before-mutation tests.
- [x] Add condensation integration tests for launch ordering, scalar/direct/
   environment temperature sources, repeated calls, float32 support, signature
   compatibility, and pre-refresh failure atomicity.
- [x] Add Warp-backed public integration tests for reusable thermodynamics,
  vapor-pressure, and mass-transfer buffers; legacy positional mass-transfer
  compatibility; and atomic missing-sidecar and CUDA cross-device failures.
- [x] Migrate benchmark and quick-start executable calls to pass the sidecar.

## Documentation

- [x] Document the validation-only and standalone refresh contracts in code and
  API docstrings.
- [ ] Cross-reference E4-F2 through E4-F7 deferred responsibilities.
