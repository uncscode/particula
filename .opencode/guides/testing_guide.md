# Testing Guide

**Project:** particula  
**Last Updated:** 2026-07-04

particula uses pytest as its primary testing framework. Tests should be close to
the code they validate and should exercise scientific correctness, edge cases,
and regression behavior.

## Framework

- **pytest:** test discovery and execution.
- **pytest-cov:** coverage reporting.
- **NumPy testing helpers:** numerical comparisons and tolerances.

## File Naming

All test files must use the `*_test.py` suffix.

```text
Correct:
  activity_coefficients_test.py
  coagulation_test.py
  vapor_pressure_test.py

Wrong:
  test_activity_coefficients.py
  activity_coefficients_tests.py
  streamTest.py
```

This pattern matters because pytest discovery, ruff per-file ignores, and agent
tooling all rely on it.

## Test Locations

Place tests in `tests/` subdirectories alongside source modules.

```text
particula/
├── activity/
│   ├── activity_coefficients.py
│   └── tests/
│       └── activity_coefficients_test.py
├── gas/
│   └── tests/
└── particles/
    └── tests/
```

Integration tests live in `particula/integration_tests/`.

## Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=particula --cov-report=term-missing

# Run a module's tests
pytest particula/activity/tests/

# Run a single file
pytest particula/activity/tests/activity_coefficients_test.py

# Run a single test
pytest particula/activity/tests/activity_coefficients_test.py::test_function_name

# Match CI warning behavior
pytest -Werror
```

## Warnings

CI treats warnings as errors with `pytest -Werror`. Tests that pass locally may
fail in CI if they emit `RuntimeWarning`, `DeprecationWarning`, or similar.

Preferred handling order:

1. Fix the underlying warning condition.
2. Use `pytest.warns()` when warning emission is intentional behavior.
3. Use a specific warning filter only when the warning is expected and not the
   subject of the test.

```python
import pytest


def test_expected_warning():
    """Test that the warning is part of the public behavior."""
    with pytest.warns(RuntimeWarning, match="radius values are zero"):
        result = function_that_warns()
    assert result is not None
```

## Scientific Test Patterns

Use `numpy.testing` for numerical comparisons.

```python
import numpy as np
import numpy.testing as npt


def test_physical_property():
    """Test a known physical-property value."""
    temperature = 298.15  # K
    pressure = 101325.0  # Pa

    result = calculate_density(temperature, pressure)
    expected = 1.184

    npt.assert_allclose(result, expected, rtol=1e-3)
```

For conservation laws, compare initial and final totals with an appropriate
tolerance.

## Performance Benchmarks

The staggered condensation benchmark suite is heavy and excluded from normal CI.
Run it manually when changing staggered condensation behavior:

```bash
pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"
```

This suite verifies O(n) scaling at 1k/10k/100k particles, theta-mode behavior,
and deterministic seeded behavior. Staggered stepping uses sequential
Gauss-Seidel updates, so high overhead compared to simultaneous vectorized
stepping is expected.

## Wall Loss Coverage

Wall loss strategy tests should cover both package export paths where relevant:

- `particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py`
- `particula/dynamics/tests/wall_loss_strategies_test.py`

Coverage should include spherical and rectangular geometry, rectangular chamber
dimension validation, supported distribution types, zero concentration, empty
particle-resolved inputs, and parity with helper functions.

## NVIDIA Warp Tests

GPU code must match Python/NumPy reference implementations. Use lightweight test
kernels around `@wp.func` functions and compare against NumPy with tight
tolerances.

```python
import numpy as np
import numpy.testing as npt
import pytest

wp = pytest.importorskip("warp")


def test_gpu_matches_numpy():
    """Warp computation matches the NumPy reference."""
    expected = numpy_reference(...)
    result = warp_result(...)
    npt.assert_allclose(result, expected, rtol=1e-10)
```

Use constants from `particula.util.constants`; do not hardcode physical
constants in kernels.

For CPU↔GPU container helpers, add round-trip coverage that checks exact value
and shape preservation on the Warp CPU backend. For `EnvironmentData`, cover
single-box and multi-box cases, the default synchronized path, any supported
manual `sync=False` path, and malformed-schema failures surfaced by CPU-side
validation.

## Test Quality

- Use descriptive test names such as `test_coagulation_conserves_total_mass`.
- Keep tests independent; do not rely on test execution order.
- Use parametrization for related input variants.
- Prefer focused assertions, but include enough checks to validate the behavior.
- Add regression tests when fixing bugs.

## Troubleshooting

- If tests are not discovered, check `*_test.py` naming and run `pytest --collect-only`.
- If imports fail, install the package in development mode with `pip install -e .[dev]`.
- If coverage looks wrong, run `pytest --cov=particula --cov-report=term`.
- If CI fails but local tests pass, rerun locally with `pytest -Werror`.
