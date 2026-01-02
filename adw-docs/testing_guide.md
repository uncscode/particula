# Testing Guide

**Version:** 0.2.6
**Last Updated:** 2025-11-30

## Overview

This guide documents all testing conventions, commands, and requirements for the **particula** repository. It serves as the single source of truth for how to write, execute, and validate tests in this codebase.

### Test Framework

particula uses **pytest** as the primary testing framework.

### Testing Toolchain

- **pytest**: Test discovery, execution, and reporting
- **pytest-cov**: Code coverage measurement
- **Standard library**: For test fixtures and assertions

### Integration with ADW

This guide is referenced by ADW (AI Developer Workflow) commands to understand repository-specific testing requirements. ADW commands use this guide to:
- Determine which test framework to use
- Know how to execute tests
- Validate test file naming and structure
- Generate coverage reports
- Resolve test failures

## Test Framework Configuration

### pytest Configuration

Tests are discovered automatically by pytest. No special configuration is needed in `pyproject.toml` for basic test discovery.

**Key Configuration:**
- **testpaths**: Tests are in `tests/` subdirectories within each module
- **Minimum test count**: 500 tests (validated by ADW)
- **Coverage**: Configured to measure coverage for `particula/` package
- **Timeout**: Test commands have a 600-second (10 minute) timeout
- **Warnings as errors**: CI runs with `-Werror` flag (see below)

### Warnings Are Treated as Errors

**IMPORTANT:** In CI, tests run with `pytest -Werror`, which treats all warnings as errors.

This means any `RuntimeWarning`, `DeprecationWarning`, or other warning will cause the test to **FAIL** in CI, even if the test logic passes.

**Why this matters:**
- Tests that pass locally may fail in CI if they emit warnings
- Warnings indicate potential issues that should be addressed
- This ensures clean, warning-free code in production

**Example failure:**
```
FAILED test_file.py::test_function - RuntimeWarning: divide by zero encountered
```

**How to handle warnings in tests:**

1. **Fix the underlying issue** (preferred): Modify code to avoid the warning condition
2. **Use `pytest.warns()` to assert expected warnings**: When a warning is intentional behavior
3. **Filter specific warnings**: When warnings are expected and acceptable

See the [Handling Expected Warnings](#handling-expected-warnings) section for details.

## File Naming Conventions

### Required Pattern: `*_test.py`

**All test files MUST use the `*_test.py` suffix.**

#### Examples

```
✓ Correct:
  stream_test.py
  activity_coefficients_test.py
  coagulation_test.py
  vapor_pressure_test.py

✗ Wrong:
  test_stream.py
  test_activity_coefficients.py
  streamTest.py
```

### Why This Pattern Matters

The `*_test.py` pattern is **critical** for:

1. **Test Discovery**: pytest automatically discovers test files matching this pattern
2. **Linting Configuration**: Ruff applies different rules to test files (e.g., allows asserts via `S101` ignore)
3. **Consistency**: Makes it easy to identify test files across the codebase
4. **Tool Integration**: ADW tools expect this naming convention

From `pyproject.toml`:
```toml
[tool.ruff.lint.per-file-ignores]
# Ignore assert-usage (S101) in any file ending with _test.py
"*_test.py" = ["S101", "E721", "B008"]
```

## Directory Structure

### Test Location: Tests Alongside Code

Tests are organized in `tests/` subdirectories within each module:

```
particula/
├── activity/
│   ├── tests/
│   │   ├── activity_coefficients_test.py
│   │   ├── bat_blending_test.py
│   │   └── ...
│   ├── activity_coefficients.py
│   └── ...
├── gas/
│   ├── tests/
│   │   ├── atmosphere_test.py
│   │   ├── species_test.py
│   │   └── ...
│   └── ...
└── particles/
    ├── tests/
    │   ├── distribution_test.py
    │   └── ...
    └── ...
```

**Benefits:**
- Tests are close to the code they test
- Easy to find related tests
- Module-level organization matches code structure
- Clear separation via `tests/` subdirectory

### Integration Tests

Integration tests are in `particula/integration_tests/`:

```
particula/
└── integration_tests/
    ├── coagulation_integration_test.py
    ├── condensation_particle_resolved_test.py
    └── quick_start_test.py
```

### Wall loss strategy coverage

Wall loss strategy tests are mirrored to ensure both package export paths work:

- `particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py`
- `particula/dynamics/tests/wall_loss_strategies_test.py`

Each file validates:

- Strategy instantiation for spherical and rectangular geometries
- Dimension validation for rectangular chambers (exactly three positive values)
- Distribution types: "discrete", "continuous_pdf", "particle_resolved"
- Aspect ratio cases (cubic, elongated, flat) produce finite rates
- Edge cases: zero concentration and empty particle-resolved inputs
- Parity with `get_rectangle_wall_loss_rate` helper functions

## Running Tests

### Basic Commands

**Run all tests:**
```bash
pytest
```

**Run with coverage:**
```bash
pytest --cov=particula --cov-report=term-missing
```

**Run tests in a specific module:**
```bash
pytest particula/activity/tests/
```

**Run a specific test file:**
```bash
pytest particula/activity/tests/activity_coefficients_test.py
```

**Run a specific test function:**
```bash
pytest particula/activity/tests/activity_coefficients_test.py::test_function_name
```

### ADW Testing Commands

**Run tests with validation (ADW tool):**
```bash
.opencode/tool/run_pytest.py
```

**Run with summary output:**
```bash
.opencode/tool/run_pytest.py --output summary
```

**Run with full output:**
```bash
.opencode/tool/run_pytest.py --output full
```

**Run with JSON output:**
```bash
.opencode/tool/run_pytest.py --output json
```

**Custom minimum test count:**
```bash
.opencode/tool/run_pytest.py --min-tests 500
```

### Performance benchmarks (slow + performance markers)

The staggered condensation performance benchmarks are heavy and excluded from CI. They
measure overhead (<2x vs simultaneous), scaling at 1k/10k/100k particles (target ~O(n)),
and compare theta modes (half, random, batch) with deterministic seeds.

**Run the suite:**
```bash
pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"
```

Notes:
- Requires `-m "slow and performance"` because the module is marked slow+performance.
- Expected overhead: staggered <2x simultaneous; scaling roughly linear in particle
  count.
- Use printed timings to compare theta modes; fixed seeds and bounded iterations reduce
  noise.

### CI/CD Commands

The GitHub Actions workflow runs:
```bash
pytest -v --tb=short --cov=particula --cov-report=term-missing
```

From `.github/workflows/test.yml`.

## Test Validation

### Minimum Test Count

ADW validates that at least **500 tests** pass to prevent false positives.

Current test count: **711 tests** (as of 2025-11-30)

If tests drop below 500, the validation will fail even if all remaining tests pass.

### Coverage Requirements

Coverage is measured with `pytest-cov` for the `particula/` package.

**Coverage command:**
```bash
pytest --cov=particula --cov-report=term-missing
```

**Coverage report format:**
- `term-missing`: Shows line numbers of missing coverage in terminal

## Writing Tests

### Test File Structure

Each test file should follow this structure:

```python
"""Tests for module_name module.

Brief description of what this test file covers.
"""

import pytest
from particula.module import function_to_test


def test_basic_functionality():
    """Test basic functionality of the function."""
    result = function_to_test(input_value)
    assert result == expected_value


def test_edge_case():
    """Test edge case handling."""
    result = function_to_test(edge_case_input)
    assert result == expected_edge_case_output


def test_error_handling():
    """Test that errors are raised appropriately."""
    with pytest.raises(ValueError):
        function_to_test(invalid_input)


class TestClassName:
    """Group related tests in a class."""

    def test_method_one(self):
        """Test method one."""
        assert True

    def test_method_two(self):
        """Test method two."""
        assert True
```

### Test Naming Conventions

**Test Functions:**
- Prefix with `test_`
- Use descriptive names: `test_calculate_density_with_zero_mass()`
- Avoid generic names: `test_1()`, `test_basic()`

**Test Classes:**
- Prefix with `Test`
- Group related tests: `TestActivityCoefficients`, `TestCoagulation`

### Assertions

Use pytest's assertion introspection (standard `assert` statements):

```python
# Good - clear and simple
assert result == expected
assert len(output) > 0
assert value in collection

# Also good - with custom messages
assert result == expected, f"Expected {expected}, got {result}"
```

### Fixtures

Use pytest fixtures for setup and teardown:

```python
import pytest
import numpy as np


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return np.array([1.0, 2.0, 3.0, 4.0])


def test_with_fixture(sample_data):
    """Test using fixture data."""
    assert len(sample_data) == 4
    assert sample_data[0] == 1.0
```

### Parametrized Tests

Use `@pytest.mark.parametrize` for testing multiple inputs:

```python
import pytest


@pytest.mark.parametrize("input_val,expected", [
    (0, 0),
    (1, 1),
    (2, 4),
    (3, 9),
])
def test_square(input_val, expected):
    """Test square function with multiple inputs."""
    assert square(input_val) == expected
```

## Common Test Patterns

### Testing NumPy Arrays

```python
import numpy as np
import numpy.testing as npt


def test_array_equality():
    """Test array equality with tolerance."""
    result = calculate_array()
    expected = np.array([1.0, 2.0, 3.0])
    npt.assert_array_almost_equal(result, expected, decimal=5)


def test_array_shape():
    """Test array shape."""
    result = calculate_matrix()
    assert result.shape == (3, 3)
```

### Testing Exceptions

```python
import pytest


def test_raises_value_error():
    """Test that function raises ValueError for invalid input."""
    with pytest.raises(ValueError, match="must be positive"):
        function_with_validation(-1)


def test_raises_type_error():
    """Test that function raises TypeError for wrong type."""
    with pytest.raises(TypeError):
        function_expecting_int("string")
```

### Handling Expected Warnings

Since CI runs with `-Werror`, tests that trigger warnings will fail. Use these patterns to handle expected warnings:

**Option 1: Assert expected warnings with `pytest.warns()`**

Use this when the warning is intentional behavior that should be tested:

```python
import pytest
import warnings


def test_expected_warning():
    """Test that function emits expected warning."""
    with pytest.warns(RuntimeWarning, match="radius values are zero"):
        result = function_that_warns()
    assert result == expected_value


def test_multiple_warnings():
    """Test function that emits multiple warnings."""
    with pytest.warns(RuntimeWarning) as warning_info:
        result = function_with_multiple_warnings()
    assert len(warning_info) == 2
    assert "first message" in str(warning_info[0].message)
```

**Option 2: Filter specific warnings with `warnings.filterwarnings`**

Use this when warnings are acceptable side effects, not the focus of the test:

```python
import warnings


def test_with_filtered_warning():
    """Test functionality while filtering known warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All radius values are zero",
            category=RuntimeWarning,
        )
        result = function_that_warns()
    assert result == expected_value
```

**Option 3: Use pytest marker to filter warnings**

Use this for test functions where warnings are expected throughout:

```python
import pytest


@pytest.mark.filterwarnings("ignore:All radius values are zero:RuntimeWarning")
def test_with_zero_radius():
    """Test with zero radius values (warning expected)."""
    result = calculate_with_zero_radius()
    assert result is not None
```

**Option 4: Configure in `pyproject.toml` (use sparingly)**

For warnings that are acceptable project-wide:

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "ignore::DeprecationWarning:some_third_party_lib.*",
]
```

**Best Practices:**
- Prefer fixing the code over suppressing warnings
- If suppressing, be as specific as possible (message + category)
- Document why a warning is being suppressed
- Use `pytest.warns()` when the warning itself is the behavior being tested

### Testing Scientific Functions

```python
import numpy as np
import numpy.testing as npt


def test_physical_property():
    """Test that physical property calculation is correct."""
    # Known test case
    temperature = 298.15  # K
    pressure = 101325.0  # Pa
    
    result = calculate_density(temperature, pressure)
    expected = 1.184  # kg/m³
    
    # Use relative tolerance for physical calculations
    npt.assert_allclose(result, expected, rtol=1e-3)


def test_conservation_law():
    """Test that conservation laws are respected."""
    initial_mass = calculate_total_mass(initial_state)
    final_mass = calculate_total_mass(final_state)
    
    # Mass should be conserved
    npt.assert_allclose(initial_mass, final_mass, rtol=1e-10)
```

## Test Organization Best Practices

### One Test, One Assertion (When Possible)

```python
# Good - focused test
def test_addition():
    """Test addition operation."""
    assert add(2, 3) == 5


# Avoid - testing multiple things
def test_all_operations():
    """Test all operations."""
    assert add(2, 3) == 5
    assert subtract(5, 3) == 2
    assert multiply(2, 3) == 6
    assert divide(6, 3) == 2
```

### Independent Tests

Each test should be independent and not rely on other tests:

```python
# Good - independent test
def test_particle_creation():
    """Test creating a particle."""
    particle = create_particle(mass=1.0)
    assert particle.mass == 1.0


def test_particle_growth():
    """Test particle growth."""
    particle = create_particle(mass=1.0)
    grown = grow_particle(particle, growth_rate=0.5)
    assert grown.mass > particle.mass


# Avoid - dependent tests (test order matters)
particle = None  # Module-level state

def test_1_create():
    global particle
    particle = create_particle(mass=1.0)
    assert particle.mass == 1.0

def test_2_grow():
    global particle  # Depends on test_1_create
    grown = grow_particle(particle, growth_rate=0.5)
    assert grown.mass > 1.0
```

### Descriptive Test Names

```python
# Good - clear what's being tested
def test_coagulation_conserves_total_mass():
    """Test that coagulation process conserves total mass."""
    pass


def test_condensation_increases_particle_size():
    """Test that condensation increases particle diameter."""
    pass


# Avoid - unclear names
def test_coag():
    """Test coagulation."""
    pass


def test_case_1():
    """Test something."""
    pass
```

## Debugging Failed Tests

### Running Tests in Verbose Mode

```bash
# Show test names as they run
pytest -v

# Show full output (stdout/stderr)
pytest -s

# Stop after first failure
pytest -x

# Show local variables on failure
pytest -l
```

### Running Specific Failed Tests

```bash
# Run last failed tests only
pytest --lf

# Run failed tests first, then others
pytest --ff
```

### Using Debugger

```python
def test_complex_calculation():
    """Test complex calculation with debugging."""
    import pdb; pdb.set_trace()  # Set breakpoint
    result = complex_function()
    assert result == expected
```

Or use pytest's built-in debugger:
```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start of test
pytest --trace
```

## Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions.

**Workflow file:** `.github/workflows/test.yml`

**Test command (CI):**
```bash
pytest -Werror
```

**IMPORTANT:** The `-Werror` flag treats all warnings as errors. This means:
- Any `RuntimeWarning`, `DeprecationWarning`, etc. will cause test failure
- Tests must be warning-free to pass in CI
- See [Handling Expected Warnings](#handling-expected-warnings) for how to address this

**Local testing with coverage:**
```bash
pytest -v --tb=short --cov=particula --cov-report=term-missing
```

**Local testing with warnings as errors (matches CI):**
```bash
pytest -Werror
```

**Requirements for PR merge:**
- All tests must pass
- No warnings emitted (due to `-Werror`)
- Minimum 500 tests must pass
- No test errors

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Make sure particula is installed in development mode
pip install -e .
```

**Missing dependencies:**
```bash
# Install development dependencies
pip install -e .[dev]
```

**Tests not discovered:**
```bash
# Check test file naming (*_test.py)
find particula -name "*test*.py"

# Run pytest with collection-only to see what's discovered
pytest --collect-only
```

**Coverage not working:**
```bash
# Install pytest-cov
pip install pytest-cov

# Verify coverage is measuring the right package
pytest --cov=particula --cov-report=term
```

## Test Metrics

**Current Status (2025-11-30):**
- Total tests: 711
- Test files: 129
- Minimum required: 500 tests
- Coverage tool: pytest-cov
- Test timeout: 600 seconds (10 minutes)

**Test Distribution by Module:**
- Activity: `particula/activity/tests/`
- Dynamics: `particula/dynamics/tests/`
- Gas: `particula/gas/tests/`
- Particles: `particula/particles/tests/`
- Util: `particula/util/tests/`
- Integration: `particula/integration_tests/`

## Summary

**Key Requirements:**
1. ✅ Use `*_test.py` naming pattern
2. ✅ Place tests in `tests/` subdirectories within modules
3. ✅ Maintain at least 500 passing tests
4. ✅ Run tests with coverage: `pytest --cov=particula`
5. ✅ Follow pytest conventions for test functions and classes
6. ✅ Write independent, focused tests with descriptive names
7. ✅ **Tests must be warning-free** (CI runs with `-Werror`)

**Quick Reference:**
```bash
# Run all tests
pytest

# Run with warnings as errors (matches CI)
pytest -Werror

# Run with coverage
pytest --cov=particula --cov-report=term-missing

# Run specific module
pytest particula/activity/tests/

# Run with ADW validation
.opencode/tool/run_pytest.py
```
