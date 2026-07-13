# Testing Strategy

Every implementation phase ships with co-located tests; coverage thresholds are
not lowered. Test files retain the `*_test.py` suffix, primarily under
`particula/gpu/kernels/tests/`.

## Per-Phase Coverage

- **P1 (shipped in #1281):** `thermodynamics_test.py` covers valid mixed
  constant/Buck sidecars, identity and buffer preservation, frozen bindings,
  missing/non-config inputs, field metadata/device/schema errors, unsupported
  modes, non-finite/negative values, ordered molar-mass mismatch, no structural
  readbacks, one-readback-per-required-buffer, and mutable-buffer revalidation.
- **P2 (shipped in #1282):** `thermodynamics_test.py` compares constant output
  with `ConstantVaporPressureStrategy` and canonical Buck output with
  `get_buck_vapor_pressure()` below, at, and above freezing. It covers one and
  multiple boxes, mixed species/models, reserved Buck parameters, complete
  overwrite, concrete-module-only export, and API validation failures that leave
  seeded vapor-pressure buffers unchanged.
- **P3 (shipped in #1283):** `condensation_test.py` and its shared support
  cover refresh launch ordering; stale-buffer overwrite and CPU parity for
  scalar, direct `wp.float64`, and `WarpEnvironmentData` temperatures; repeated
  temperature changes with reused gas/configuration; direct `wp.float32`
  temperature casting; and legacy positional/signature compatibility. Parameterized
  pre-refresh failure regressions assert malformed thermodynamics, invalid
  physical inputs, invalid optional mass-transfer buffers, and device mismatch
  raise before a refresh launch or gas/particle mutation.
- **Condensation boundary (shipped in #1281):** Regression tests require the
  keyword-only sidecar and prove invalid/missing configurations fail before
  launch, helper/scratch allocation, mass-transfer access, or mutation of
  particle mass, gas concentration, vapor pressure, or caller outputs. Existing
  single/multi-box parity and stiffness paths pass a valid sidecar unchanged.
- **P5:** Validate documentation links, mode/units tables, and cross-references.

## Device and Numerical Policy

- Warp CPU parity is required when Warp is installed; CUDA is optional and skips
  cleanly when unavailable.
- Use explicit `float64` dtype/shape, `assert_allclose` tolerances for formula
  parity, and exact buffer snapshots for failure-before-mutation assertions.
- Focused verification covers `thermodynamics_test.py`, condensation and
  stiffness tests, including `-k "refresh or temperature or pre_refresh"`, and
  the opt-in benchmark; CUDA validation remains optional.
- Maintain at least 80% changed-code coverage and never lower repository gates.
