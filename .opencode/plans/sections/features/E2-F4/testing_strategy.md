# Testing Strategy

## Co-Located Testing Policy

Each phase ships with tests for the behavior it changes. There is no standalone
testing phase.

## Unit and Conversion Tests

- `E2-F4-P1`, `E2-F4-P2`, and `E2-F4-P3` use focused coverage in
  `particula/gpu/tests/conversion_test.py`.
- Added and maintained focused coverage for:
  - CPU-to-Warp gas conversion shapes and dtypes;
  - bool→`int32` and `int32`→bool `partitioning` conversion;
  - explicit and omitted `vapor_pressure` handling;
  - invalid vapor-pressure shape errors with actual/expected shape text;
  - supplied names, placeholder-name fallback, explicit `name=None`, and
    name-length mismatch failures;
  - empty provided name-list failures with actual/expected count messaging;
  - non-binary `partitioning` restore failures with `partitioning` + binary
    `0/1` message assertions;
  - retry-safe recovery after correcting invalid names or invalid
    `partitioning` values;
  - GPU-only `vapor_pressure` presence before restore and loss after restore.
- `E2-F4-P3` pairs those tests with the vapor-pressure contract wording in
  `particula/gpu/conversion.py` and `particula/gpu/warp_types.py`.

## Integration and Documentation Checks

- Run GPU conversion tests on Warp CPU where available; do not require CUDA for
  basic semantic validation.
- Run focused tests for changed modules first, then the normal fast suite.
- Verify docs match the exact behavior asserted by tests.

## Acceptance Test Matrix

| Behavior | Expected Coverage |
| --- | --- |
| Names supplied to `from_warp_gas_data()` | Preserved exactly |
| Names omitted or `None` | Placeholder-name restore contract |
| Wrong-length or empty provided names | `ValueError` with actual/expected count text |
| Partitioning round trip | CPU bool preserved through GPU int32 for binary `0/1` inputs |
| Invalid restored partitioning | `ValueError` before bool coercion |
| Vapor pressure supplied | Shape validated and values transferred to Warp |
| Vapor pressure omitted | Zero-filled `(n_boxes, n_species)` GPU default |
| Invalid vapor-pressure shape | `ValueError` with actual/expected shape text |
| Vapor pressure returned to CPU | Intentionally discarded on `GasData` restore, with sidecar preservation required |
