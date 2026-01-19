# Linting Guide

**Version:** 0.2.6
**Last Updated:** 2025-12-03

## Overview

This guide documents all linting conventions, commands, and requirements for the **particula** repository. It serves as the single source of truth for code quality standards, linting tools, and how to ensure code passes all CI linting checks.

### Linting Approach

particula uses **ruff (check + format) and mypy** to ensure comprehensive code quality:

1. **ruff check** - Fast Python linter with auto-fix capabilities
2. **ruff format** - Fast Python formatter
3. **mypy** - Static type checking for Python (REQUIRED)

**Why this approach?**

Ruff is a modern, extremely fast linter written in Rust that combines the functionality of multiple traditional Python linting tools into a single tool. Combined with mypy for strict type checking, this provides comprehensive code quality enforcement while keeping CI times fast.

**All linters are required** - both ruff and mypy must pass with zero errors for code to be accepted.

### Integration with ADW

This guide is referenced by ADW (AI Developer Workflow) commands to understand repository-specific linting requirements. ADW commands use this guide to:
- Determine which linters to run and in what order
- Know which issues can be auto-fixed vs require manual intervention
- Understand target directories and exclusions
- Validate that code meets quality standards before commits

## Linter Configuration

### 1. Ruff (Linting and Formatting)

**Purpose:** Modern Python linter and formatter with auto-fix capabilities.

**Installation:**
```bash
pip install ruff
# Or as part of dev dependencies
pip install -e .[dev]
```

**Commands:**

Linting with auto-fix:
```bash
ruff check particula/ --fix
```

Formatting:
```bash
ruff format particula/
```

Check without fixing:
```bash
ruff check particula/
```

**Configuration** (`pyproject.toml`):
```toml
[tool.ruff]
line-length = 80
fix = true
extend-exclude = [
  "**/*.ipynb",    # ignore every .ipynb anywhere in the project
]

[tool.ruff.format]
docstring-code-line-length = 80

[tool.ruff.lint]
select = [
  "E", "F", "W", "C90", "D", "ANN", "B", "S", "N", "I"
]
ignore = [
  "D203",  # one-blank-line-before-class (conflicts with D211)
  "D205",  # blank-line-after-summary
  "D213",  # multi-line-summary-second-line (conflicts with D212)
  "D417",  # Missing argument descriptions in the docstring
]
extend-ignore = [
  "ANN",   # ignore all missing-type-*/missing-return-type checks
]

[tool.ruff.lint.per-file-ignores]
# Ignore assert-usage (S101) in any file ending with _test.py
"*_test.py" = ["S101", "E721", "B008"]

[tool.ruff.lint.pydocstyle]
# enforce Google-style sections and disable incompatible rules
convention = "google"
```

**Key Settings:**
- **Line length**: 80 characters
- **Docstring convention**: Google-style
- **Auto-fix**: Enabled by default
- **Test files**: Allow asserts (S101), type comparisons (E721), and function calls in defaults (B008)

**Selected Rules:**
- `E`: pycodestyle errors
- `F`: Pyflakes
- `W`: pycodestyle warnings
- `C90`: mccabe complexity
- `D`: pydocstyle (docstrings)
- `ANN`: type hint annotations
- `B`: bugbear (likely bugs and design problems)
- `S`: security checks (bandit rules)
- `N`: naming conventions
- `I`: import sorting

**Note:** Rule category names reference traditional tools (flake8-*, pep8-*, etc.) but all checks are implemented natively in ruff. We do not use separate flake8, pylint, or other legacy linters.

### 2. Mypy (Type Checking) - REQUIRED

**Purpose:** Static type checker for Python - ensures type safety across the codebase.

**Status:** **REQUIRED** - All code must pass mypy type checking with zero errors.

**Installation:**
```bash
pip install mypy
# Or as part of dev dependencies
pip install -e .[dev]
```

**Commands:**

Run type checking:
```bash
mypy particula/ --ignore-missing-imports
```

**Configuration:**

Currently configured via command-line arguments. Can be added to `pyproject.toml` if needed:

```toml
[tool.mypy]
ignore_missing_imports = true
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
```

**Key Settings:**
- **Ignore missing imports**: Enabled (for dependencies without type stubs)
- **Python version**: 3.12+ (matches project requirement)

**Type Checking Standards:**
- All functions must have proper type hints for parameters and return values
- Use `Union[float, NDArray[np.float64]]` for functions that accept both scalars and arrays
- Use `Optional[T]` for parameters that can be `None`
- Add type narrowing with `isinstance()` checks before operations that require specific types
- Use `cast()` from `typing` module when type assertions are needed

## Running Linters

### Quick Commands

**Run all linters (ADW way):**
```bash
# Following CI workflow: fix → format → check
.opencode/tool/run_linters.py
```

**Run manually (following CI workflow):**
```bash
# Step 1: Apply fixes
ruff check particula/ --fix

# Step 2: Format code
ruff format particula/

# Step 3: Final check (this determines pass/fail)
ruff check particula/

# Step 4: Type check
mypy particula/ --ignore-missing-imports
```

**Run individual linters:**
```bash
# Ruff check only
ruff check particula/

# Ruff format only
ruff format particula/

# Mypy only
mypy particula/ --ignore-missing-imports
```

### ADW Linter Tool

**Run with summary output:**
```bash
.opencode/tool/run_linters.py --output summary
```

**Run with full output:**
```bash
.opencode/tool/run_linters.py --output full
```

**Run with JSON output:**
```bash
.opencode/tool/run_linters.py --output json
```

**Disable auto-fix:**
```bash
.opencode/tool/run_linters.py --no-auto-fix
```

**Run specific linters:**
```bash
.opencode/tool/run_linters.py --linters ruff
.opencode/tool/run_linters.py --linters mypy
.opencode/tool/run_linters.py --linters ruff,mypy
```

**Custom target directory:**
```bash
.opencode/tool/run_linters.py --target-dir particula/activity/
```

### CI/CD Linting

The GitHub Actions workflow runs:

```bash
# Step 1: Apply fixes (don't fail if issues found)
ruff check particula/ --fix

# Step 2: Format code
ruff format particula/

# Step 3: Final check (fail if issues remain)
ruff check particula/

# Step 4: Type check (REQUIRED - fail if errors found)
mypy particula/ --ignore-missing-imports
```

From `.github/workflows/lint.yml`.

**All linters must pass** - CI will fail if either ruff or mypy reports errors.

## Auto-Fix Capabilities

### What Ruff Can Auto-Fix

Ruff can automatically fix many issues:

**Import Sorting:**
```python
# Before
import os
import numpy as np
from particula import Aerosol
import sys

# After (auto-fixed)
import os
import sys

import numpy as np

from particula import Aerosol
```

**Trailing Whitespace:**
```python
# Before
def function():    
    return True    

# After (auto-fixed)
def function():
    return True
```

**Missing/Extra Blank Lines:**
```python
# Before
def func1():
    pass
def func2():
    pass

# After (auto-fixed)
def func1():
    pass


def func2():
    pass
```

**Unused Imports:**
```python
# Before
import os
import sys
import numpy as np

def use_numpy():
    return np.array([1, 2, 3])

# After (auto-fixed)
import numpy as np

def use_numpy():
    return np.array([1, 2, 3])
```

**Code Formatting (via ruff format):**
- Line length enforcement (80 characters)
- Consistent indentation
- Quote normalization
- Trailing commas in multi-line structures

### What Requires Manual Fixing

Some issues cannot be auto-fixed and require manual intervention:

**Complexity Issues:**
```python
# C901: Function too complex (cyclomatic complexity > threshold)
def complex_function(x, y, z):
    if x:
        if y:
            if z:
                # ... many nested conditions
                pass
# Fix: Refactor to reduce complexity
```

**Security Issues:**
```python
# S608: Possible SQL injection
query = f"SELECT * FROM users WHERE id = {user_id}"
# Fix: Use parameterized queries
query = "SELECT * FROM users WHERE id = ?"
```

**Type Issues (Mypy):**
```python
# error: Incompatible return value type
def get_value() -> int:
    return "string"  # Wrong type
# Fix: Return correct type or fix type annotation
```

**Missing Docstrings:**
```python
# D103: Missing docstring in public function
def calculate_density(mass, volume):
    return mass / volume
# Fix: Add docstring
```

## Common Linting Issues

### Import Sorting (I001)

**Issue:**
```python
import numpy as np
from particula import Aerosol
import os
```

**Fix:**
```bash
ruff check particula/ --fix  # Auto-fixes
```

**Result:**
```python
import os

import numpy as np

from particula import Aerosol
```

### Unused Variables (F841)

**Issue:**
```python
def calculate():
    result = expensive_calculation()
    other = another_calculation()  # F841: Unused variable
    return result
```

**Fix:**
```python
def calculate():
    result = expensive_calculation()
    _ = another_calculation()  # Or remove if truly unused
    return result
```

### Line Too Long (E501)

**Issue:**
```python
def function_with_very_long_name(parameter1, parameter2, parameter3, parameter4):
    pass
```

**Fix (via ruff format):**
```python
def function_with_very_long_name(
    parameter1, parameter2, parameter3, parameter4
):
    pass
```

### Missing Docstring (D103)

**Issue:**
```python
def calculate_density(mass: float, volume: float) -> float:
    return mass / volume
```

**Fix:**
```python
def calculate_density(mass: float, volume: float) -> float:
    """Calculate density from mass and volume.
    
    Args:
        mass: Mass in kg.
        volume: Volume in m³.
        
    Returns:
        Density in kg/m³.
    """
    return mass / volume
```

### Assert in Non-Test Code (S101)

**Issue:**
```python
# In production code
def process(data):
    assert len(data) > 0  # S101: Use assert only for debugging
    return process_data(data)
```

**Fix:**
```python
def process(data):
    if len(data) == 0:
        raise ValueError("Data must not be empty")
    return process_data(data)
```

**Note:** Asserts are allowed in test files (`*_test.py`).

## Linter Workflow Integration

### Pre-commit Hooks

particula has a `.pre-commit-config.yaml` that can run linters automatically:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

**Enable pre-commit hooks:**
```bash
pip install pre-commit
pre-commit install
```

Now linters run automatically on `git commit`.

### IDE Integration

**VS Code:**

Install the Ruff extension and add to `.vscode/settings.json`:
```json
{
  "ruff.enable": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll": true,
    "source.organizeImports": true
  }
}
```

**PyCharm:**

Ruff can be configured as an external tool or via the File Watcher feature.

### ADW Workflow Integration

The linter agent runs as part of the `complete` workflow:

1. **Plan** phase: Creates implementation plan
2. **Build** phase: Implements changes
3. **Lint** phase: Runs `.opencode/tool/run_linters.py` ← YOU ARE HERE
4. **Test** phase: Runs tests
5. **Review** phase: Code review
6. **Document** phase: Updates documentation
7. **Ship** phase: Creates PR

## Troubleshooting

### Linter Not Found

```bash
# Install ruff
pip install ruff

# Or install all dev dependencies
pip install -e .[dev]
```

### Configuration Not Applied

```bash
# Verify ruff sees your config
ruff check --show-settings particula/

# Check for conflicting configs
find . -name "pyproject.toml" -o -name "ruff.toml" -o -name ".ruff.toml"
```

### Auto-Fix Not Working

```bash
# Ensure fix is enabled
ruff check particula/ --fix

# Some issues can't be auto-fixed - run without --fix to see them
ruff check particula/
```

### Mypy Import Errors

```bash
# Install missing type stubs
pip install types-<package-name>

# Or ignore missing imports (already in command)
mypy particula/ --ignore-missing-imports
```

## Summary

**Key Requirements:**
1. ✅ Run `ruff check --fix` to auto-fix issues
2. ✅ Run `ruff format` to format code
3. ✅ Run `ruff check` to verify (final check)
4. ✅ Run `mypy` for type checking (**REQUIRED** - must pass with zero errors)
5. ✅ Follow Google-style docstrings
6. ✅ Keep lines to 80 characters
7. ✅ Use `*_test.py` files for tests (asserts allowed)

**Quick Reference:**
```bash
# Run all linters with ADW tool
.opencode/tool/run_linters.py

# Or manually (CI workflow)
ruff check particula/ --fix
ruff format particula/
ruff check particula/
mypy particula/ --ignore-missing-imports  # REQUIRED

# Target directory: particula/
# Line length: 80
# Docstring style: Google
# Test files: *_test.py (asserts allowed)
# Type checking: Required (mypy must pass)
```

**Linting Tools:**
- **ruff**: Modern Python linter and formatter (replaces flake8, pylint, black, isort)
- **mypy**: Static type checker (required for all code)
