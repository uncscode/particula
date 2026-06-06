# Linting Guide

**Version:** 2.1.0
**Last Updated:** 2025-11-14

## Overview

This guide documents all linting conventions, commands, and requirements for the adw repository. It serves as the single source of truth for code quality standards, linting tools, and how to ensure code passes all CI linting checks.

### Linting Approach

adw uses **ruff and mypy** to ensure comprehensive code quality:

1. **ruff** - Modern Python linter and formatter with auto-fix capabilities
2. **mypy** - Static type checker for Python code

**Why ruff and mypy?**

- **ruff**: Fast, modern Python linter that replaces flake8, isort, pydocstyle, and more. Supports auto-fixing many issues.
- **mypy**: Catches type-related bugs before runtime through static type analysis.

This combination provides comprehensive code quality checking with minimal configuration.

### Integration with ADW

This guide is referenced by ADW (Agent Developer Workflow) commands to understand repository-specific linting requirements. ADW commands use this guide to:
- Determine which linters to run and in what order
- Know which issues can be auto-fixed vs require manual intervention
- Understand target directories and exclusions
- Validate that code meets quality standards before commits

### `run_linters` Tool Contract

The OpenCode `run_linters` tool exposes two distinct modes:

- `autoFix: false` is **validation-only** and **non-mutating**. It runs Ruff
  checks without `--fix` and does not run `ruff format`.
- `autoFix: true` is the **mutating** path. It follows the CI-style Ruff flow:
  `ruff check --fix`, `ruff format`, then a final validation check, stopping
  early if any Ruff step fails.

Use `autoFix: false` when you need a read/check pass that must leave the
worktree unchanged.

## Linter Configuration

### Linter 1: ruff

**Purpose:** Modern Python linter and formatter with auto-fix capabilities

**Commands:**

Linting with auto-fix:
```bash
ruff check adw/ --fix
```

Formatting:
```bash
ruff format adw/
```

Check formatting without modifying:
```bash
ruff format --check adw/
```

Final check (CI validation):
```bash
ruff check adw/
```

**Auto-fix:** Yes - Many issues can be automatically fixed

**What it checks:**
- **E**: pycodestyle errors (PEP 8 violations)
- **F**: pyflakes (unused imports, undefined names)
- **W**: pycodestyle warnings
- **I**: isort (import sorting)
- **N**: pep8-naming (naming conventions)

**Configuration** (`pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
exclude = [
    "templates/",
    "trees/",
    ".*",         # Exclude all dotfiles and dot directories at root
    "**/.*",      # Exclude dotfiles and dot directories in subdirectories
]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N"]
```

### Linter 2: mypy

**Purpose:** Static type checker for Python

**Command:**
```bash
mypy adw/ --ignore-missing-imports
```

**Auto-fix:** No - Type errors must be fixed manually

**What it checks:**
- Type annotation correctness
- Type compatibility in assignments
- Function argument types
- Return type mismatches
- Optional/None handling

**Configuration:** Uses default mypy settings with `--ignore-missing-imports` flag to skip type checking for third-party libraries without type stubs.

## Line Length and Formatting

### Line Length Limit: 100 Characters

adw enforces a **100-character line length limit**.

**Why 100 characters?**

- Modern monitors can easily display 100 characters
- Balances readability with code density
- Allows side-by-side diffs and split-screen editing
- Aligns with many modern Python projects (e.g., Black's default)

### Automatic Formatting

**ruff** automatically formats code to meet the line length limit:

```bash
ruff format adw/
```

This command reformats all Python files in the `adw/` directory to comply with the 100-character limit and other formatting standards.

## Target Directories and Exclusions

### Primary Target

**All linters target the `adw` directory.**

This directory contains all application code, tests, and modules.

### Excluded Directories

The following directories and files are excluded from linting (configured in `pyproject.toml`):

- **`templates/`** - Template files for project initialization
- **`trees/`** - Git worktree directories for isolated ADW workflows
- **`.*`** - All dotfiles and dot directories (e.g., `.github/`, `.claude/`, `.git/`, `.env`)
- **`__pycache__/`** - Python bytecode cache (automatically excluded by ruff)
- **`.venv/`** - Virtual environment (automatically excluded by ruff)

**Note:** Test files (`*_test.py`) are linted with the same rules as production code. The `adw/tests/` directory is not excluded.

## Linting Workflow

### Recommended Order

Run linters in this order for most efficient workflow (matches CI/CD pipeline):

1. **ruff check --fix** (auto-fix issues first)
   ```bash
   ruff check adw/ --fix
   ```
   Fixes import sorting, removes unused imports, and corrects many style issues automatically.
   Note: CI runs this with `|| true` to prevent early failure, allowing formatting to proceed.

2. **ruff format** (auto-format code)
   ```bash
   ruff format adw/
   ```
   Automatically fixes formatting issues like line length, indentation, and spacing.

3. **ruff check** (final validation)
   ```bash
   ruff check adw/
   ```
   Validates that all linting rules pass after fixes. This is what CI checks for success.

4. **mypy** (type check)
   ```bash
   mypy adw/ --ignore-missing-imports
   ```
   Validates type annotations and catches type-related errors (cannot auto-fix).

### Run All Linters Sequentially (CI Pipeline Order)

```bash
ruff check adw/ --fix || true && \
ruff format adw/ && \
ruff check adw/ && \
mypy adw/ --ignore-missing-imports
```

This command runs all linters in the exact order used by CI, ensuring local validation matches CI validation.

## Pre-commit Hooks

### Purpose

Pre-commit hooks automatically run linting checks before each git commit, preventing commits with linting errors.

### Configuration

Pre-commit hooks are configured in `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      # Run the linter with auto-fix
      - id: ruff
        args: [--fix]
        exclude: ^templates/projects/
      # Run the formatter
      - id: ruff-format
        exclude: ^templates/projects/
```

### Installation

```bash
pre-commit install
```

### Manual Execution

Run pre-commit checks on all files:
```bash
pre-commit run --all-files
```

Run pre-commit checks on staged files only:
```bash
pre-commit run
```

**Note:** Pre-commit hooks exclude `templates/projects/` directory to prevent modifying language-specific templates.

## Quick Reference

### Most Common Commands

```bash
# Lint with auto-fix (run first)
ruff check adw/ --fix

# Format code
ruff format adw/

# Validate linting (final check)
ruff check adw/

# Type check
mypy adw/ --ignore-missing-imports

# Check formatting without modifying
ruff format --check adw/

# Run all linters (CI pipeline order)
ruff check adw/ --fix || true && ruff format adw/ && ruff check adw/ && mypy adw/ --ignore-missing-imports
```

### Linter Comparison Table

| Linter | Auto-fix | Target | Config File |
|--------|----------|--------|-------------|
| ruff check | Yes | `adw/` | `pyproject.toml` |
| ruff format | Yes | `adw/` | `pyproject.toml` |
| mypy | No | `adw/` | Command-line flags |

## Troubleshooting

### Issue 1: Linting Errors After Auto-fix

**Symptom:** Linter reports errors after running auto-fix command.

**Cause:** Some issues cannot be auto-fixed and require manual intervention (e.g., unused variables, complex naming issues).

**Solution:** Review error messages and fix manually. Common manual fixes:
- Remove unused variables
- Rename variables to follow naming conventions
- Fix logical errors flagged by linter

### Issue 2: Type Checking Errors

**Symptom:** mypy reports type errors.

**Cause:** Type annotations are missing, incorrect, or incompatible.

**Solution:**
- Add type annotations to function signatures
- Fix type mismatches (e.g., assigning str to int variable)
- Use `Optional[T]` for values that can be None
- Use `Union[T1, T2]` for values that can be multiple types

### Issue 3: Import Sorting Conflicts

**Symptom:** ruff reorders imports, creating merge conflicts.

**Cause:** Multiple developers format code differently before committing.

**Solution:**
- Always run `ruff check adw/ --fix` before committing
- Use pre-commit hooks to enforce consistent import sorting
- Resolve conflicts by accepting ruff's import order

## See Also

### Configuration Files

- **`pyproject.toml`**: ruff configuration (line length, rules, exclusions)
- **`.pre-commit-config.yaml`**: Pre-commit hook configuration

### Related Documentation

- **`.opencode/guides/testing_guide.md`**: Testing conventions (test files have special linting rules)
- **`.opencode/guides/code_style.md`**: Code style conventions enforced by linters
- **`.opencode/guides/commit_conventions.md`**: Commit message conventions

### External Resources

- **ruff Documentation**: https://docs.astral.sh/ruff/
- **mypy Documentation**: https://mypy.readthedocs.io/
- **PEP 8 Style Guide**: https://peps.python.org/pep-0008/
- **PEP 484 Type Hints**: https://peps.python.org/pep-0484/

---

**Questions or issues?** Consult project documentation or open an issue.
