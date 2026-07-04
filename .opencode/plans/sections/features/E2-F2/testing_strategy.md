# Testing Strategy

Every implementation phase ships with co-located tests. Coverage thresholds are
not lowered.

## Per-Phase Testing

- **P1: Fields and validation**
  - File: `particula/gas/tests/environment_data_test.py`
  - Test valid single-box construction.
  - Test invalid dimensionality for each field.
  - Test mismatched `(n_boxes,)` and `(n_boxes, n_species)` lengths.
  - Test non-finite temperature/pressure/`saturation_ratio` values.
  - Test invalid negative values and positive-Kelvin requirements.
  - Use parametrized cases that prove `saturation_ratio > 1.0` remains valid while
    negative values still fail.

- **P2: Container API and exports**
  - File: `particula/gas/tests/environment_data_test.py`
  - Test valid multi-box construction.
  - Test dtype coercion to `np.float64`.
  - Test `n_boxes` property.
  - Test `copy()` returns independent arrays.
  - Test import from `particula.gas`.

- **P3: Documentation**
  - Validate docs by running available markdown/link tooling or at minimum
    checking changed references manually.
  - No production-code tests are added in this phase unless docs examples are
    introduced.

## Commands

- Scoped tests: `pytest particula/gas/tests/environment_data_test.py -q`
- Broader gas regression: `pytest particula/gas/tests -q`
- Lint touched code: `ruff check particula/gas --fix && ruff format particula/gas`
- Full validation when practical: `pytest` and repository lint workflow.

## Coverage Impact

The new container should have direct unit coverage for all branches in
`__post_init__`, `n_boxes`, exports, and `copy()`. This feature should increase
or preserve package coverage.
