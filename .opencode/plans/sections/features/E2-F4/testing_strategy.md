# Testing Strategy

## Co-Located Testing Policy

Each phase ships with tests for the behavior it changes. There is no standalone
testing phase.

## Unit and Conversion Tests

- `E2-F4-P1` extended `particula/gpu/tests/conversion_test.py` only.
- Added focused coverage for:
  - CPU-to-Warp gas conversion shapes and dtypes;
  - bool→`int32` and `int32`→bool `partitioning` conversion;
  - explicit and omitted `vapor_pressure` handling;
  - invalid vapor-pressure shape errors with actual/expected shape text;
  - supplied names, placeholder-name fallback, and name-length mismatch
    failures;
  - GPU-only `vapor_pressure` presence before restore and loss after restore.
- No `warp_types`, `gas_data`, or production-code tests were needed because the
  landed phase was contract-locking coverage rather than a runtime change.

## Integration and Documentation Checks

- Run GPU conversion tests on Warp CPU where available; do not require CUDA for
  basic semantic validation.
- Run focused tests for changed modules first, then the normal fast suite.
- Verify docs match the exact behavior asserted by tests.

## Acceptance Test Matrix

| Behavior | Expected Coverage |
| --- | --- |
| Names supplied to `from_warp_gas_data()` | Preserved exactly |
| Names omitted | Placeholder-name restore contract |
| Partitioning round trip | CPU bool preserved through GPU int32 |
| Vapor pressure supplied | Shape validated and values transferred to Warp |
| Vapor pressure omitted | Zero-filled `(n_boxes, n_species)` GPU default |
| Vapor pressure returned to CPU | Intentionally discarded on `GasData` restore |
