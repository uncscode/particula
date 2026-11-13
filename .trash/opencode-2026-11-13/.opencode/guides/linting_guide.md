# Linting Guide

**Project:** particula  
**Last Updated:** 2026-06-06

particula uses ruff and mypy for code quality. Both should pass before code is
considered ready.

## Tools

- **ruff check:** linting and import sorting, with auto-fix support.
- **ruff format:** code formatting with 80-character line length.
- **mypy:** static type checking.

## Standard Workflow

Run linters in this order:

```bash
ruff check particula/ --fix
ruff format particula/
ruff check particula/
mypy particula/ --ignore-missing-imports
```

The final `ruff check` and `mypy` runs determine whether lint validation passes.

## Ruff Configuration

Important settings from `pyproject.toml`:

```toml
[tool.ruff]
line-length = 80
fix = true
extend-exclude = [
  "**/*.ipynb",
]

[tool.ruff.format]
docstring-code-line-length = 80

[tool.ruff.lint]
select = ["E", "F", "W", "C90", "D", "ANN", "B", "S", "N", "I"]
ignore = ["D203", "D205", "D213", "D417"]
extend-ignore = ["ANN"]

[tool.ruff.lint.per-file-ignores]
"*_test.py" = ["S101", "E721", "B008"]

[tool.ruff.lint.pydocstyle]
convention = "google"
```

Key implications:

- Keep lines to 80 characters.
- Use Google-style docstrings.
- Test files named `*_test.py` may use `assert`.
- Missing type annotation ruff rules are ignored, but public type hints remain
  expected by repository convention.

## Mypy

Run mypy with missing imports ignored:

```bash
mypy particula/ --ignore-missing-imports
```

Type-checking expectations:

- Add parameter and return types for public functions.
- Use `NDArray[np.float64]` for NumPy arrays where dtype matters.
- Use type narrowing before operations that require a specific type.
- Use `cast()` only when the runtime invariant is clear.

## Common Fixes

### Import Sorting

```bash
ruff check particula/ --fix
```

Ruff sorts imports into standard-library, third-party, and local groups.

### Formatting

```bash
ruff format particula/
```

Ruff handles line wrapping, quote normalization, indentation, and trailing
commas.

### Missing Docstrings

Public modules, classes, and functions should have Google-style docstrings.
Include units and citations where scientific context matters.

### Asserts

Use assertions freely in `*_test.py` files. Do not use `assert` for runtime
validation in production code; raise explicit exceptions instead.

## Pre-Commit

If pre-commit is installed, enable hooks with:

```bash
pre-commit install
```

Run all hooks manually with:

```bash
pre-commit run --all-files
```

## Troubleshooting

- If ruff settings look wrong, run `ruff check --show-settings particula/`.
- If tests are not treated as test files, confirm the filename ends in `_test.py`.
- If mypy reports missing third-party stubs, keep using `--ignore-missing-imports`
  unless the project adds explicit stubs.

## Quick Reference

```bash
ruff check particula/ --fix
ruff format particula/
ruff check particula/
mypy particula/ --ignore-missing-imports
```
