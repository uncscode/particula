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
  - Shipped tests cover `n_boxes` for the validated per-box container API.
  - Shipped tests prove `copy()` returns independent arrays with
    `np.shares_memory(...)` checks.
  - Shipped tests verify copy-mutation isolation for `temperature`, `pressure`,
    and `saturation_ratio`.
  - Shipped tests verify import from `particula.gas` and retain smoke coverage
    for the direct-module path.

- **P3: Documentation**
  - Shipped validation is documentation-only: re-read the changed feature-guide
    and roadmap sections, confirm they agree on ownership and mutation
    boundaries, and inspect changed references manually.
  - Preferred tooling remains `mkdocs build --strict` when available, but no
    production-code tests are added in this phase.

## Commands

- Scoped tests: `pytest particula/gas/tests/environment_data_test.py -q`
- Broader gas regression: `pytest particula/gas/tests -q`
- Lint touched code: `ruff check particula/gas --fix && ruff format particula/gas`
- Full validation when practical: `pytest` and repository lint workflow.

## Coverage Impact

The new container already has direct unit coverage for `__post_init__`, the
validation helpers, `n_boxes`, exports, and `copy()` without weakening the
existing constructor contract. The shipped documentation phase does not add new
runtime coverage and should not reduce package coverage.
