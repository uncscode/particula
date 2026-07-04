# Implementation Tasks

## E2-F4-P1: Audit Current Contract

- Inspect `GasData`, `WarpGasData`, and gas conversion helpers after `E2-F1`
  lands.
- Add tests that lock down current field ownership and round-trip outcomes.
- Ensure tests include names, molar mass, concentration, partitioning, vapor
  pressure, and invalid vapor-pressure shapes.
- Update docstrings only where they are clearly stale.

## E2-F4-P2: Name and Partitioning Semantics

- Implement `from_warp_gas_data()` so CPU restoration prefers caller-supplied
  `name` input or documented external metadata, without treating `WarpGasData`
  as authoritative name storage.
- If placeholder names remain as a compatibility fallback, keep that path
  explicit in `particula/gpu/conversion.py` docstrings and tests rather than as
  silent metadata preservation.
- Add tests in `particula/gpu/tests/conversion_test.py` for supplied names,
  missing names, mismatched name counts, and any placeholder fallback behavior.
- Add tests that GPU `int32` partitioning returns CPU boolean partitioning and
  that invalid or non-binary values raise before casting.

## E2-F4-P3: Vapor-Pressure Semantics

- Treat `vapor_pressure` as explicit process or GPU-helper state rather than a
  field owned by CPU `GasData` or `EnvironmentData`.
- Keep `to_warp_gas_data(..., vapor_pressure=...)` shape validation explicit for
  `(n_boxes, n_species)` buffers in `particula/gpu/conversion.py`.
- Document and test the accepted CPU-return behavior: callers who need vapor
  pressure after GPU execution must read it from `WarpGasData` before
  `from_warp_gas_data()` or manage an explicit sidecar.
- Add tests for supplied, missing, invalid-shape, and intentional non-round-trip
  vapor-pressure behavior.

## E2-F4-P4: Documentation

- Update migration docs with a field-by-field CPU/GPU authority table.
- Update roadmap notes so known schema drift is described as intentional or
  resolved.
- Include migration guidance for users with name-keyed downstream logic.
- Include guidance for computing and passing vapor pressure to GPU kernels.
