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
  - Required validation path: run `mkdocs build --strict` when command
    execution is available to the builder.
  - Manual re-read and reference inspection are fallback-only validation when
    the docs build cannot be run in the available execution environment.
  - Any fallback note must explicitly record why `mkdocs build --strict` was
    unavailable for the fix pass.
  - No new production-code tests are required for this phase unless claim
    verification reveals shipped behavior without existing supporting coverage.

## Behavior-Claim Verification Checklist

- Package export claim:
  confirm `particula/gas/__init__.py` imports and exports `EnvironmentData`
  for `from particula.gas import EnvironmentData`.
- Ownership claim:
  confirm `particula/gas/environment_data.py` defines `EnvironmentData` as the
  CPU owner of per-box `temperature`, `pressure`, and `saturation_ratio`.
- Copy-semantics claim:
  confirm `particula/gas/environment_data.py` implements `copy()` with
  independent NumPy arrays for all three fields.
- Regression-coverage claim:
  confirm `particula/gas/tests/environment_data_test.py` covers the package
  export path plus copy-value, copy-memory-independence, and copy-mutation
  isolation behavior.
- Compatibility-boundary claim:
  confirm changed docs preserve the statement that existing process APIs may
  still accept scalar `temperature` and `pressure` until later migrations.

## Commands

- Scoped tests: `pytest particula/gas/tests/environment_data_test.py -q`
- Broader gas regression: `pytest particula/gas/tests -q`
- Lint touched code: `ruff check particula/gas --fix && ruff format particula/gas`
- Full validation when practical: `pytest` and repository lint workflow.

## Coverage Impact

The new container already has direct unit coverage for `__post_init__`, the
validation helpers, `n_boxes`, exports, and `copy()` without weakening the
existing constructor contract. The shipped documentation phase does not add new
runtime coverage by default, should not reduce package coverage, and must
verify behavior claims against the shipped code/tests above.
