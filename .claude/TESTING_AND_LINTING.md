# Testing and Linting Guide for Particula

This guide covers pytest testing and ruff linting/formatting for the Particula project.

## Pytest - Testing Framework

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests in a specific directory
uv run pytest particula/activity/tests/

# Run a specific test file
uv run pytest particula/activity/tests/activity_coefficients_test.py

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=particula --cov-report=html

# Run tests matching a pattern
uv run pytest -k "test_activity"
```

### Test Structure and Conventions

**Test File Naming:**
- Test files are named with `*_test.py` suffix (e.g., `activity_coefficients_test.py`)
- Located in `tests/` subdirectories alongside source code
- Example: `particula/activity/tests/activity_coefficients_test.py`

**Test Function Naming:**
- Test functions start with `test_` prefix
- Use descriptive names: `test_activity_coefficients()`, `test_mass_condensation()`

**Test Organization:**
```python
def test_function_name():
    """Test for function_name function."""
    # Arrange - Set up test data
    input_data = np.array([1.0, 2.0, 3.0])

    # Act - Call the function
    result = function_name(input_data)

    # Assert - Verify expected behavior
    assert np.all(result >= 0)
    assert result.shape == input_data.shape
```

**Common Assertions:**
- `assert value == expected` - Exact equality
- `np.allclose(a, b)` - Floating point comparison with tolerance
- `np.all(condition)` - All elements meet condition
- `assert isinstance(obj, ExpectedType)` - Type checking
- `pytest.raises(ExceptionType)` - Exception testing

**Example Test Pattern:**
```python
def test_bat_activity_coefficients():
    """Test for BAT activity coefficients function."""
    # Setup
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2500

    # Execute
    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        organic_density=density
    )

    # Verify
    assert np.all(activity_coefficients[0] >= 0)
    assert activity_coefficients[0].shape == (10,)
```

### Integration Tests

Located in: `particula/integration_tests/`

**Key Integration Tests:**
- `quick_start_test.py` - End-to-end workflow example
- `coagulation_integration_test.py` - Coagulation process testing
- `condensation_particle_resolved_test.py` - Condensation testing

### Test Configuration

Pytest configuration is in `pyproject.toml`. Tests run automatically via GitHub Actions on push/PR.

---

## Ruff - Fast Python Linter and Formatter

### Running Ruff

**Linting:**
```bash
# Lint all files
uv run ruff check .

# Lint specific directory
uv run ruff check particula/

# Lint with auto-fix
uv run ruff check --fix .

# Lint specific file
uv run ruff check particula/activity/activity_coefficients.py
```

**Formatting:**
```bash
# Format all files
uv run ruff format .

# Format specific directory
uv run ruff format particula/

# Check formatting without making changes
uv run ruff format --check .

# Format specific file
uv run ruff format particula/activity/activity_coefficients.py
```

### Ruff Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 80
indent-width = 4

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "F",      # pyflakes
    "W",      # pycodestyle warnings
    "C90",    # mccabe complexity
    "D",      # pydocstyle (docstrings)
    "ANN",    # flake8-annotations
    "B",      # flake8-bugbear
    "S",      # flake8-bandit (security)
    "N",      # pep8-naming
    "I",      # isort (import sorting)
]

[tool.ruff.lint.pydocstyle]
convention = "google"  # Google-style docstrings
```

### Code Style Rules

**Line Length:** 80 characters maximum

**Docstring Convention:** Google style
```python
def function_name(param1: float, param2: np.ndarray) -> float:
    """Short description of function.

    Longer description if needed, explaining what the function does,
    its purpose, and any important details.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.

    Examples:
        >>> result = function_name(1.0, np.array([1, 2, 3]))
        >>> print(result)
        6.0
    """
    # Implementation
    pass
```

**Import Sorting:**
```python
# Standard library imports
import os
from pathlib import Path

# Third-party imports
import numpy as np
from scipy import constants

# Local imports
from particula.util.constants import BOLTZMANN_CONSTANT
```

### Per-File Ignores

Test files have relaxed rules (configured in `pyproject.toml`):

```toml
[tool.ruff.lint.per-file-ignores]
"*test*.py" = [
    "S101",   # Allow assert in tests
    "E721",   # Allow type comparisons in tests
    "B008",   # Allow function calls in argument defaults
    "ANN",    # Skip type annotations in tests
    "D",      # Skip docstring requirements in tests
]
```

### Common Linting Issues and Fixes

**Issue: Line too long**
```python
# Bad
result = some_very_long_function_name(parameter1, parameter2, parameter3, parameter4)

# Good
result = some_very_long_function_name(
    parameter1, parameter2, parameter3, parameter4
)
```

**Issue: Missing docstring**
```python
# Bad
def calculate_mass(radius, density):
    return (4/3) * np.pi * radius**3 * density

# Good
def calculate_mass(radius: float, density: float) -> float:
    """Calculate mass from radius and density.

    Args:
        radius: Particle radius in meters.
        density: Particle density in kg/m³.

    Returns:
        Particle mass in kg.
    """
    return (4/3) * np.pi * radius**3 * density
```

**Issue: Import not sorted**
```python
# Bad
from particula.util.constants import BOLTZMANN_CONSTANT
import numpy as np
import os

# Good (ruff format --fix will fix this automatically)
import os

import numpy as np

from particula.util.constants import BOLTZMANN_CONSTANT
```

---

## GitHub Actions CI/CD

### Automated Testing

Tests run automatically via `.github/workflows/test.yml`:
```yaml
- name: Test with pytest
  run: pytest
```

### Automated Linting

Linting runs via `.github/workflows/lint.yml`:
```yaml
- name: Lint with ruff
  run: ruff check .
```

---

## Best Practices

### Before Committing

Run both checks locally:
```bash
# 1. Format code
uv run ruff format .

# 2. Lint code
uv run ruff check --fix .

# 3. Run tests
uv run pytest

# 4. Check test coverage (optional)
uv run pytest --cov=particula
```

### Writing New Tests

1. **Create test file** in appropriate `tests/` directory
2. **Name with** `*_test.py` suffix
3. **Import pytest** and necessary modules
4. **Write test functions** starting with `test_`
5. **Use descriptive assertions** with clear error messages
6. **Test edge cases**: empty inputs, zero values, negative values, boundary conditions
7. **Test error handling**: Use `pytest.raises()` for expected exceptions

### Writing New Code

1. **Add type hints** to function signatures
2. **Write docstrings** using Google style
3. **Keep functions short** (< 50 lines preferred)
4. **Follow naming conventions**: snake_case for functions/variables, PascalCase for classes
5. **Line length**: 80 characters max
6. **Add unit tests** for new functions
7. **Run ruff format** before committing

---

## Quick Reference

```bash
# Run all checks (recommended before commit)
uv run ruff format . && uv run ruff check --fix . && uv run pytest

# Run specific test file
uv run pytest particula/path/to/test_file.py -v

# Run tests with pattern matching
uv run pytest -k "condensation" -v

# Check formatting without changing files
uv run ruff format --check .

# Lint with auto-fix
uv run ruff check --fix .

# Run tests with coverage
uv run pytest --cov=particula --cov-report=term-missing
```

---

## Troubleshooting

**Tests failing locally but pass in CI:**
- Check Python version (CI runs on Python 3.10–3.14)
- Ensure all dependencies installed: `uv sync`
- Check for platform-specific issues

**Ruff formatting conflicts with editor:**
- Configure editor to use ruff for formatting
- Disable other formatters (autopep8, black, etc.)
- Set line length to 80 in editor settings

**Import errors in tests:**
- Install package in editable mode: `uv sync`
- Check PYTHONPATH includes project root
- Verify `__init__.py` files exist in package directories

**Coverage issues:**
- Exclude test files from coverage
- Focus on testing public APIs, not internal helpers
- Aim for >80% coverage on critical modules
