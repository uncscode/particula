# Success Criteria

## Functional Criteria

- `EnvironmentData` exists as a CPU dataclass in the gas/data-container area of
  the package.
- The container stores `temperature` and `pressure` as `np.float64` arrays
  shaped `(n_boxes,)`, plus `saturation_ratio` as an `np.float64` array shaped
  `(n_boxes, n_species)`.
- Single-box construction keeps the leading box dimension (`(1,)` and
  `(1, n_species)`), and multi-box construction supports more than one box.
- Invalid dimensionality, mismatched box counts, non-finite values, nonpositive
  pressure, negative saturation ratio, and invalid temperature values raise
  clear `ValueError`s.
- Supersaturation values above `1.0` remain valid and are covered as an
  explicit accepted case.
- P1 shipped the direct-module import path
  `particula.gas.environment_data.EnvironmentData`.
- Later phases still need to add `n_boxes`, `copy()`, and the package export
  path.

## Test Criteria

- New unit tests cover valid single-box and multi-box inputs.
- New unit tests cover invalid shapes, species-dimension mismatches, and
  invalid values.
- New unit tests cover dtype coercion.
- Later phases still need test coverage for copy independence and package
  exports.
- New unit tests include at least one supersaturation case (`saturation_ratio >
  1.0`) and one nonpositive-pressure validation failure.
- Scoped gas tests pass.

## Documentation Criteria

- Documentation explains that `EnvironmentData` owns per-box thermodynamic
  state.
- Documentation explains how processes should read and mutate environment state
  and notes that process API migrations are downstream.

## Done Signal

Issue #1188 satisfied the P1 done signal: `EnvironmentData` exists with the
canonical `temperature`, `pressure`, and species-resolved `saturation_ratio`
fields, plus tests for valid and invalid construction. Full feature completion
still requires the later `n_boxes`, `copy()`, export, and documentation phases.
