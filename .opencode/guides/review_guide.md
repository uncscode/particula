# Review Guide

**Project:** particula  
**Last Updated:** 2026-06-06

Review particula changes for scientific correctness first, then behavior,
maintainability, tests, and documentation.

## Validation Commands

```bash
pytest
pytest --cov=particula --cov-report=term-missing
ruff check particula/ --fix
ruff format particula/
ruff check particula/
mypy particula/ --ignore-missing-imports
```

Run targeted tests for small changes and broader tests for cross-module or
scientific-model changes.

## Review Checklist

- Scientific formulas match source references and preserve units.
- Numerical methods use appropriate tolerances and avoid avoidable warnings.
- Public functions validate inputs with `validate_inputs` where applicable.
- NumPy vectorization is used where practical for array-heavy calculations.
- Edge cases are covered, including zeros, empty arrays, invalid inputs, and
  scalar-versus-array behavior.
- Tests are co-located in `tests/` directories and named `*_test.py`.
- CI-warning behavior is respected; code and tests should pass under `pytest -Werror`.
- New public APIs have type hints and Google-style docstrings.
- Documentation or examples are updated when user-facing behavior changes.

## Performance-Sensitive Code

For condensation, coagulation, wall-loss, or GPU/Warp changes, check algorithmic
complexity and memory behavior. Avoid Python loops over large arrays unless the
method is intentionally sequential, such as staggered Gauss-Seidel stepping.

## Wall Loss Changes

Wall loss changes should preserve both direct module behavior and exported
package behavior. Review spherical and rectangular chamber paths, distribution
types, zero concentration handling, and helper-function parity.

## Notebook Changes

For `docs/Examples/` notebooks, review the paired `.py` source and `.ipynb`.
The `.py` should be linted, synced to the notebook, and the notebook executed
when feasible.
