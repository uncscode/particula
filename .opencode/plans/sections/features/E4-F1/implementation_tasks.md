# Implementation Tasks

## GPU Physics and API

- [x] Define stable `wp.int32` constant and Buck mode codes in the concrete
  thermodynamics module.
- [x] Add a frozen, caller-owned fixed-shape configuration with typed,
  species-indexed Warp fields.
- [x] Validate required configuration, supported modes, finite/nonnegative
  values, dtype, shape, species ordering, and active Warp device.
- [ ] Implement constant and piecewise Buck Warp calculations from the exact CPU
  references using `float64`.
- [ ] Implement a refresh launch that writes every `(box, species)` output slot.
- [x] Add required keyword-only `thermodynamics` to `condensation_step_gpu()`.
- [x] Invoke validation after active-context setup and before defaults,
  mass-transfer access, allocation, or launch.
- [x] Ensure all validation failures occur before `gas.vapor_pressure`, gas
  concentration, or particle mass can mutate.

## Tooling / Tests

- [x] Add focused validator unit tests in
  `particula/gpu/kernels/tests/thermodynamics_test.py`.
  or a focused `*_test.py` beside the new module.
- [ ] Add mixed-model, multi-box, freezing-boundary, repeated-temperature, and
  species-order tests.
- [x] Add invalid/missing configuration and failure-before-mutation tests.
- [x] Migrate benchmark and quick-start executable calls to pass the sidecar.

## Documentation

- [x] Document the validation-only contract in code and API docstrings.
- [ ] Cross-reference E4-F2 through E4-F7 deferred responsibilities.
