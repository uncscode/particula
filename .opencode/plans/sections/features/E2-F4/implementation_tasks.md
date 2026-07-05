# Implementation Tasks

## E2-F4-P1: Audit Current Contract

- Landed in issue `#1197` as a focused test-only update in
  `particula/gpu/tests/conversion_test.py`.
- Locked current field ownership and round-trip outcomes for names, molar mass,
  concentration, `partitioning`, vapor pressure, and invalid vapor-pressure
  shapes.
- Added coverage for placeholder-name fallback, empty/wrong-length name
  failures, and GPU-only `vapor_pressure` loss on CPU restore.
- Left docstrings and production semantics unchanged in this phase.

## E2-F4-P2: Name and Partitioning Semantics

- Landed in issue `#1198` through `particula/gpu/conversion.py`,
  `particula/gpu/warp_types.py`, and `particula/gpu/tests/conversion_test.py`.
- `from_warp_gas_data()` now prefers caller-supplied `name` input or documented
  external metadata, without treating `WarpGasData` as authoritative name
  storage.
- Placeholder names remain an explicit documented fallback rather than silent
  metadata preservation.
- Focused tests cover supplied names, missing names, mismatched name counts,
  and binary/non-binary `partitioning` restore behavior.

## E2-F4-P3: Vapor-Pressure Semantics

- Landed in issue `#1199` through `particula/gpu/conversion.py`,
  `particula/gpu/warp_types.py`, and `particula/gpu/tests/conversion_test.py`.
- Kept `vapor_pressure` as explicit GPU-helper state rather than adding it to
  CPU `GasData` or `EnvironmentData`.
- Documented `to_warp_gas_data(..., vapor_pressure=...)` as optional caller
  input with required `(n_boxes, n_species)` validation and zero-filled default
  allocation when omitted.
- Documented and tested the accepted CPU-return behavior: callers who need
  vapor pressure after GPU execution must read it from `WarpGasData` before
  `from_warp_gas_data()` or manage an explicit sidecar.
- Added direct tests for supplied, omitted, invalid-shape, and intentionally
  lossy restore behavior.

## E2-F4-P4: Documentation

- Update migration docs with a field-by-field CPU/GPU authority table.
- Update roadmap notes so known schema drift is described as intentional or
  resolved.
- Include migration guidance for users with name-keyed downstream logic.
- Include guidance for computing and passing vapor pressure to GPU kernels.
