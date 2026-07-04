# Success Criteria

## Functional Criteria

- `EnvironmentData` exists as a CPU dataclass in the gas/data-container area of
  the package.
- The container stores per-box temperature, pressure, and humidity/saturation
  state as `np.float64` arrays shaped `(n_boxes,)`.
- Single-box construction uses shape `(1,)` and multi-box construction supports
  more than one box.
- Invalid shapes, mismatched box counts, non-finite values, and invalid
  negative or out-of-range values raise clear `ValueError`s.
- `n_boxes` and `copy()` behavior match existing data-container conventions.
- The class is importable from the expected package export path.

## Test Criteria

- New unit tests cover valid single-box and multi-box inputs.
- New unit tests cover invalid shapes and invalid values.
- New unit tests cover dtype coercion, copy independence, and package exports.
- Scoped gas tests pass.

## Documentation Criteria

- Documentation explains that `EnvironmentData` owns per-box thermodynamic
  state.
- Documentation explains how processes should read and mutate environment state
  and notes that process API migrations are downstream.

## Done Signal

EnvironmentData exists with per-box thermodynamic fields and tests for valid and
invalid shapes, satisfying issue #1172 track T2.
