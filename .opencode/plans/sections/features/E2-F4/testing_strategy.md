# Testing Strategy

## Co-Located Testing Policy

Each phase ships with tests for the behavior it changes. There is no standalone
testing phase.

## Unit and Conversion Tests

- Extend `particula/gpu/tests/conversion_test.py` for:
  - CPU-to-Warp gas conversion with supplied vapor pressure.
  - Invalid vapor-pressure shape errors.
  - Missing vapor-pressure behavior, whether default, warning, or error.
  - Warp-to-CPU conversion with supplied names.
  - Missing-name behavior, whether placeholder, warning, or error.
  - Invalid name count errors.
  - `partitioning` bool-to-int32 and int32-to-bool round trips.
- Extend `particula/gpu/tests/warp_types_test.py` only if struct docs or shape
  assumptions change.
- Extend `particula/gas/tests/gas_data_test.py` if `GasData` ownership or
  docstring-visible semantics change.

## Integration and Documentation Checks

- Run GPU conversion tests on Warp CPU where available; do not require CUDA for
  basic semantic validation.
- Run focused tests for changed modules first, then the normal fast suite.
- Verify docs match the exact behavior asserted by tests.

## Acceptance Test Matrix

| Behavior | Expected Coverage |
| --- | --- |
| Names supplied to `from_warp_gas_data()` | Preserved exactly |
| Names omitted | Explicit placeholder, warning, or error contract |
| Partitioning round trip | CPU bool preserved through GPU int32 |
| Vapor pressure supplied | Shape validated and values transferred to Warp |
| Vapor pressure omitted | Explicit documented behavior |
| Vapor pressure returned to CPU | Preserved sidecar or intentionally discarded by test |
