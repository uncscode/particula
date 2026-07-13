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
- **P2:** Compare Warp CPU results with `ConstantVaporPressureStrategy` and
  `get_buck_vapor_pressure()` below, at, and above freezing. Cover one/multiple
  boxes, one/multiple species, and mixed model ordering.
- **P3:** Deferred; no formula or vapor-pressure-refresh behavior was added.
- **Condensation boundary (shipped in #1281):** Regression tests require the
  keyword-only sidecar and prove invalid/missing configurations fail before
  launch, helper/scratch allocation, mass-transfer access, or mutation of
  particle mass, gas concentration, vapor pressure, or caller outputs. Existing
  single/multi-box parity and stiffness paths pass a valid sidecar unchanged.
- **P5:** Validate documentation links, mode/units tables, and cross-references.

## Device and Numerical Policy

- Warp CPU parity is required when Warp is installed; CUDA is optional and skips
  cleanly when unavailable.
- Use exact dtype/shape and buffer snapshot assertions. Formula parity and
  freezing-boundary tests remain deferred with formula implementation.
- Focused verification covers `thermodynamics_test.py`, condensation and
  stiffness tests, and the opt-in benchmark; CUDA validation remains optional.
- Maintain at least 80% changed-code coverage and never lower repository gates.
