# Testing Strategy

Every implementation phase ships with co-located tests. Coverage thresholds are
not lowered.

## Per-Phase Testing

- **P1: Fields and validation**
  - File: `particula/gas/tests/environment_data_test.py`
  - Shipped tests cover valid single-box and multi-box construction.
  - Shipped tests verify coercion from lists/tuples to `np.float64` arrays.
  - Test invalid dimensionality for each field.
  - Test mismatched `(n_boxes,)` and `(n_boxes, n_species)` lengths.
  - Test non-finite temperature/pressure/`saturation_ratio` values.
  - Test nonpositive pressure, negative saturation_ratio values, and
    positive-Kelvin requirements.
  - Use parametrized cases that prove `saturation_ratio > 1.0` remains valid while
    negative values still fail.
  - Shipped tests also exercise private validation helpers so future phases can
    refactor with confidence, but public-constructor coverage remains the main
    contract.

- **P2: Container API and exports**
  - File: `particula/gas/tests/environment_data_test.py`
  - Test `n_boxes` property.
  - Test `copy()` returns independent arrays.
  - Test import from `particula.gas`.
  - Retain a smoke test confirming direct-module import still works after export
    wiring is added.

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

The new container already has direct unit coverage for `__post_init__` and the
current validation helpers. Remaining phases should add targeted coverage for
`n_boxes`, exports, and `copy()` without weakening the existing constructor
contract. This feature should increase or preserve package coverage.
