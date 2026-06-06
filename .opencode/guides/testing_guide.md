# Testing Guide

**Version:** 2.3.0
**Last Updated:** 2026-05-28

## Overview

This guide documents all testing conventions, commands, and requirements for the adw repository. It serves as the single source of truth for how to write, execute, and validate tests in this codebase.

### Out of Scope

- **Plan artifacts**: Structured plan content under `.opencode/plans/` (JSON metadata, phase details, section markdown) is not in scope for testing. Plans are living documents that change as work progresses; writing tests that assert on plan content creates brittle coupling to transient project-management state. Do not write tests that validate plan phase statuses, plan markdown content, or plan-to-doc consistency.

### Test Framework

adw uses **pytest** as the primary testing framework.

### Testing Toolchain

- **pytest**: Test discovery, execution, and reporting (>=7.4)
- **pytest-cov**: Code coverage measurement (>=4.1)
- **pytest-xdist**: In-runner parallel pytest execution for CI and local speedups
- **pytest-asyncio**: Async test support (>=0.21)
- **unittest.mock**: Mocking for external dependencies

### Integration with ADW

This guide is referenced by ADW (Agent Developer Workflow) commands to understand repository-specific testing requirements. ADW commands use this guide to:
- Determine which test framework to use
- Know how to execute tests
- Validate test file naming and structure
- Generate coverage reports
- Resolve test failures

## Test Framework and Tools

### pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["adw", ".opencode/tools/tool_tests"]
python_files = "*_test.py"
addopts = "-v --import-mode=importlib -m 'not slow' --cov=adw --cov-fail-under=80 --cov-report=term-missing"
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

**Key Configuration:**
- **testpaths**: Tests are discovered in the `adw` directory
- **python_files**: Test files must match the `*_test.py` suffix pattern
- **addopts**: Verbose output, importlib mode, default coverage, and local `not slow` filtering
- **filterwarnings**: Deprecation warnings are ignored
- **Timeout**: Test commands have a 2 minute timeout (extendable to 10 minutes for complex tests)

## File Naming Conventions

### Required Pattern: `*_test.py`

**All test files MUST follow the `*_test.py` suffix naming convention.**

This pattern is configured in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
python_files = "*_test.py"
```

#### Correct Examples:
```
✓ agent_test.py                    # Tests for agent.py module
✓ context_test.py                  # Tests for context.py module
✓ health_test.py                   # Tests for health utilities
✓ workflow_operations_test.py      # Tests for workflow operations
✓ github_operations_test.py        # Tests for GitHub operations
✓ guide_references_test.py         # Integration test for guide references
✓ portability_compliance_test.py   # Integration test for portability
✓ label_validation_test.py         # Test for label validation logic
```

#### Wrong Examples:
```
✗ test_agent.py          # Wrong: test_ prefix instead of _test suffix
✗ agent.test.py          # Wrong: dot separator instead of underscore
✗ agent_tests.py         # Wrong: plural "tests" instead of "test"
✗ testagent.py           # Wrong: no separator between "test" and module name
```

### Why File Naming Matters

The `*_test.py` suffix pattern is **critical** for:

1. **Test Discovery**: pytest automatically discovers test files matching this pattern
2. **Linting Configuration**: ruff applies specific rules to test files (e.g., allows assert statements)
3. **Coverage Configuration**: Coverage tool omits `*_test.py` files from coverage reports (configured in `pyproject.toml`)
4. **Team Conventions**: Consistent naming aids navigation and understanding
5. **Clear Separation**: The suffix pattern makes it immediately obvious which files are tests vs implementation

**Naming Convention:**
- Use descriptive names that indicate what is being tested (e.g., `agent_test.py` tests `agent.py`, `workflow_operations_test.py` tests workflow operations)
- For integration tests, use descriptive names that indicate the integration being tested (e.g., `guide_references_test.py`, `portability_compliance_test.py`)

## Directory Structure Requirements

### Test Location

**Tests Alongside Code** - This repository uses a modular structure with tests placed in `tests/` subdirectories within each module:

```
adw/
├── core/
│   ├── __init__.py
│   ├── agent.py
│   ├── context.py
│   ├── models.py
│   └── tests/              # Tests for core module
│       ├── agent_test.py
│       ├── context_test.py
│       └── models_test.py
├── github/
│   ├── __init__.py
│   ├── client.py
│   ├── operations.py
│   └── tests/              # Tests for github module
│       ├── client_test.py
│       └── github_operations_test.py
├── workflows/
│   ├── __init__.py
│   ├── plan.py
│   ├── build.py
│   └── tests/              # Tests for workflows module
│       ├── workflows_test.py
│       └── dispatcher_test.py
└── tests/                  # Top-level integration tests
    ├── integration_workflow_test.py
    ├── guide_references_test.py
    └── portability_compliance_test.py
```

### Special Requirements

- **No `__init__.py` in test directories**: Test directories should NOT contain `__init__.py` files to prevent them from being treated as installable packages
- **One test directory per module**: Each module has its own `tests/` subdirectory containing tests for that module only
- **Top-level tests directory**: `adw/tests/` contains integration tests and cross-module validation tests

## Test Execution Commands

### Full Test Suite

#### Run All Tests

```bash
pytest
```

This is the primary command for validating the adw package. It runs the configured testpaths, skips `slow` tests by default, and includes terminal coverage reporting.

#### Run All Tests with Verbose Output

```bash
pytest -v
```

Verbose mode is the default (configured in `pyproject.toml`), but you can explicitly specify it with `-v`.

#### Add Extra Coverage Reports

```bash
pytest --cov-report=html --cov-report=xml
```

Coverage is already enabled by default. Add extra report formats when you need HTML or XML artifacts in addition to the terminal report.

### Targeted Test Execution

#### Run Specific Test File

```bash
pytest adw/core/tests/agent_test.py
```

Run tests from a single file.

#### Run Tests for Specific Module

```bash
pytest adw/core/tests/
pytest adw/github/tests/
pytest adw/workflows/tests/
```

Run all tests for a specific module.

#### Run Single Test Case

```bash
pytest adw/core/tests/agent_test.py::test_generate_slash_command_model_map_with_real_commands
```

Run a specific test function by name.

#### Symlink-Mode Verification Sweep (E23-F3 closeout)

For `.opencode` destination symlink-mode validation and trusted-root behavior,
run this focused suite:

```bash
pytest adw/git/tests/worktree_test.py -v
pytest adw/commands/tests/pull_opencode_test.py -v
(cd .opencode && bun test tools/__tests__/run_pytest_basic.test.ts)
```

Interpretation guidance:

- Treat any failure as a blocker for docs closeout until triaged.
- Separate environment/toolchain failures from behavior regressions in your
  notes.
- Report partial execution explicitly (do not mark full verification complete
  when one surface was skipped).

### Pre-Test Validation Commands

Before running tests, it's recommended to run these validation commands:

#### 1. Syntax Validation
```bash
python -m py_compile adw/**/*.py
```
**Purpose**: Validates Python syntax by compiling to bytecode, catching syntax errors like missing delimiters or invalid indentation.

#### 2. Code Quality Check (Linting)
```bash
ruff check adw/
```
**Purpose**: Validates code quality, identifies unused imports, style violations, and potential bugs using ruff linter.

#### 3. Code Formatting Check
```bash
ruff format --check adw/
```
**Purpose**: Checks that code is properly formatted according to project standards (100 character line length).

#### 4. Type Checking
```bash
mypy adw/ --ignore-missing-imports
```
**Purpose**: Validates type annotations and catches type-related errors using mypy.

#### 5. Notebook Validation (structure, syntax, sync)
```bash
validate_notebook notebooks/ --recursive --output-mode json
```
**Purpose**: Validates notebook JSON/schema, code cell syntax (after stripping `%` magics and skipping `%%` cells), and Jupytext sync status. Exit codes: 0 (valid/in-sync), 1 (validation/sync failure), 2 (tool error). Hidden paths and `.ipynb_checkpoints/` are skipped automatically.

#### 6. Agent Reference Validation (OpenCode instructions)
Preferred validator-agent path:

```python
run_validate_agent_references({
    "cwd": "<worktree_path>"
})
```

Manual/CI shell path:

```bash
scripts/validate_agent_references.sh
# Or run directly:
python scripts/validate_agent_references.py
```
**Purpose**: Ensures `@path` and `filePath` references plus broad-wrapper policy
checks pass across the validator inventory scope: `.opencode/agent/*.md`,
`.opencode/workflow/*.json`, `.opencode/tools/*.md`, and `AGENTS.md`.
Use `run_validate_agent_references` only from the two allowlisted agents
(`docs-validator`, `adw-validate`); it is root-scoped and refuses to run when
`scripts/validate_agent_references.py` has local uncommitted edits. Keep the
shell/Python commands for CI and manual operator execution.

#### 7. Docs/Wrapper Policy Validation (docs-only minimum)
```bash
pytest adw/tests/guide_references_test.py -v
pytest adw/tests/portability_compliance_test.py -v
pytest adw/tests/agent_permission_validation_test.py -v
pytest adw/tests/agent_reference_validation_test.py -v
pytest adw/tests/agent_worktree_cwd_guidance_test.py -v
pytest adw/tests/plan_research_drafter_agent_test.py -v
pytest adw/tests/plan_research_drafter_docs_test.py -v
pytest adw/tests/wrapper_archive_consistency_test.py -v
pytest adw/tests/tool_wrapper_exceptions_metadata_test.py -v
(cd .opencode && bun test tools/__tests__)
scripts/validate_agent_references.sh
```
**Purpose**: Covers doc-link integrity, broad-wrapper policy classification,
agent permission/reference constraints, wrapper archive/exception metadata
contracts, and stable docs/guidance policy markers used by operator-facing
documentation checks.

For validator-oriented agent runs, substitute the dedicated
`run_validate_agent_references` wrapper for the shell-script step above.

#### Custom Agent Definition Tests

Tests for custom agent markdown files should validate only header/frontmatter requirements, such as `mode`, allowed tools, and other structured metadata. Do not assert on the agent body text or prompt wording; agent bodies are implementation guidance and should be free to evolve without breaking tests.

Stable repository docs/guidance contract tests are a separate category: they may
assert durable policy markers (for example wrapper-policy or worktree/cwd
guardrails) when those markers protect active workflow contracts.

### Full Test Execution Sequence

The complete test validation sequence (as used in CI):

```bash
# 1. Install dependencies
uv pip install --system -e ".[dev]"

# 2. Notebook validation (fail fast on corruption or sync issues)
validate_notebook notebooks/ --recursive --output-mode json
validate_notebook notebooks/ --check-sync --recursive

# 3. Agent reference validation (CI/manual shell path)
scripts/validate_agent_references.sh

# Validator-oriented agent path:
# run_validate_agent_references({"cwd": "<worktree_path>"})

# 4. Linting
ruff check adw/
ruff format --check adw/

# 5. Type checking
mypy adw/ --ignore-missing-imports

# 6. Tests (coverage included by default)
pytest --cov-report=html --cov-report=xml
```

### Critical Test Validation for Automated Workflows

**IMPORTANT**: When running tests in automated workflows (like ADW test workflow), you MUST verify test results thoroughly to prevent false positives:

#### Required Validation Steps

1. **Capture Full Output**: Use `tee` or redirect stderr+stdout to capture complete pytest output
2. **Check Exit Code**: Pytest must return 0 (any non-zero indicates failure)
3. **Search for FAILED Marker**: Grep output for "FAILED" - if found, tests failed even if exit code is 0
4. **Verify Test Count**: Ensure the number of passing tests matches expectations (>=1600 for this project)
5. **Check for Test Collection Errors**: Look for "ERROR" in collection phase
6. **Verify Critical Test Files Ran**: Confirm key test files (e.g., `agent_test.py`) were executed

#### Example Robust Test Validation Script

```bash
# Run pytest with full output capture
pytest -v --tb=short 2>&1 | tee pytest_output.txt

# Store exit code immediately (before tee completes)
exit_code=${PIPESTATUS[0]}

# 1. Check exit code
if [ $exit_code -ne 0 ]; then
    echo "ERROR: pytest exited with code $exit_code"
    tail -100 pytest_output.txt  # Show last 100 lines for context
    exit 1
fi

# 2. Check for FAILED marker (catches individual test failures)
if grep -q " FAILED " pytest_output.txt; then
    echo "ERROR: Some tests failed:"
    grep " FAILED " pytest_output.txt
    exit 1
fi

# 3. Verify test count (adjust 1600 based on your test suite size)
test_count=$(grep -oP '\d+(?= passed)' pytest_output.txt | tail -1)
if [ -z "$test_count" ]; then
    echo "ERROR: Could not parse test count from output"
    exit 1
fi
if [ "$test_count" -lt 1600 ]; then
    echo "ERROR: Expected at least 1600 tests, but only $test_count passed"
    echo "This indicates tests were not fully run or many were skipped"
    exit 1
fi

# 4. Check for ERROR in test collection
if grep -q "ERROR collecting" pytest_output.txt; then
    echo "ERROR: Test collection failed:"
    grep "ERROR collecting" pytest_output.txt
    exit 1
fi

# 5. Verify critical test file was included
if ! grep -q "agent_test.py" pytest_output.txt; then
    echo "WARNING: Critical test file agent_test.py may not have run"
fi

echo "SUCCESS: All $test_count tests passed with full validation"
```

#### Why This Matters

Without these checks, you may encounter:
- **Truncated output**: Agent sees partial output, misses failures at the end
- **Silent failures**: Tests fail but exit code is incorrectly 0 due to shell issues
- **Skipped tests**: Major test files skipped due to import errors or collection issues
- **False positives**: Agent reports "all tests passed" when many actually failed

#### Integration with ADW /test Command

The `/test` slash command should use this robust validation to ensure test results are accurate before marking the test phase as complete.

**Recommended**: Use the provided validation script for comprehensive test checking:

```bash
# Run tests with full validation
./scripts/validate_tests.sh

# Or with specific pytest options
./scripts/validate_tests.sh --cov=adw --cov-report=term-missing
```

This script (`scripts/validate_tests.sh`) implements all the validation steps above and provides clear pass/fail reporting.

## Coverage Requirements

### Coverage Commands

#### Default Terminal Coverage Report

```bash
pytest
```

**Output**: Shows coverage percentage per module with line numbers of uncovered code in the terminal because coverage is part of the default pytest configuration.

#### HTML Coverage Report

```bash
pytest --cov-report=html
```

**Output**: Generates detailed HTML report in `htmlcov/` directory.
**View Report**: Open `htmlcov/index.html` in a browser to see detailed per-file coverage with highlighted uncovered lines.

#### XML Coverage Report (for CI/Codecov)

```bash
pytest --cov-report=xml
```

**Output**: Generates `coverage.xml` file for upload to Codecov or other coverage tracking services.

#### Full Coverage Report (All Formats)

```bash
pytest --cov-report=html --cov-report=xml
```

This generates all common report formats and still fails if coverage drops below 80%.

### Coverage Configuration

Coverage is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
omit = [
    "*/tests/*",
    "*_test.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
fail_under = 80
```

**Coverage Omissions:**
- Test files (`*_test.py`) are excluded from coverage
- Test directories (`*/tests/*`) are excluded

**Excluded Lines:**
- Lines marked with `# pragma: no cover`
- `__repr__` methods
- Assertion and NotImplementedError raises
- Main blocks and type checking blocks

### Coverage Thresholds and Interpretation

Coverage reports show the percentage of code executed during tests:

- **High coverage (≥80%)**: Most code paths are tested, good confidence in code quality
- **Medium coverage (60-79%)**: Adequate testing, but gaps exist that may hide bugs
- **Low coverage (<60%)**: Significant untested code, higher risk of undetected issues

### Coverage Enforcement

adw requires a minimum coverage of **80%**. CI will fail if coverage drops below this threshold (`--cov-fail-under=80` in CI, `fail_under = 80` in `pyproject.toml`).

## Test Duration Tiers

Tests should be categorized by execution time to enable fast feedback during development while still supporting comprehensive validation in CI/CD pipelines.

### Duration Tiers

| Tier | Duration | Usage | Marker |
|------|----------|-------|--------|
| **Fast** | ≤ 1 second | Unit tests, isolated module tests | (no marker needed) |
| **Slow** | > 1 second | Integration tests, complex setups | `@pytest.mark.slow` |
| **Performance** | Up to 5 minutes | Benchmarks, load tests, stress tests | `@pytest.mark.performance` |

### Guidelines by Tier

#### Fast Tests (≤ 1 second) - Default

The majority of tests should be fast. These run on every commit and provide immediate feedback.

**Characteristics:**
- Isolated unit tests
- Mocked external dependencies
- No network calls, no filesystem I/O (or minimal)
- Test single functions or small modules

**Example:**
```python
def test_validate_input_basic():
    """Test basic input validation - should complete in milliseconds."""
    result = validate_input("valid_data")
    assert result is True
```

#### Slow Tests (>1 second) - `@pytest.mark.slow`

Tests that require more setup time or integration with multiple components. Slow tests are
excluded from the default CI marker expression and are commonly run in dedicated suites.

**Characteristics:**
- Integration tests across multiple modules
- Tests requiring complex fixtures or setup
- Tests with retries or timeouts
- Database or filesystem operations

**Example:**
```python
@pytest.mark.slow
def test_workflow_integration():
    """Test complete workflow execution - may take several seconds."""
    # Complex setup with multiple components
    workflow = create_test_workflow()
    result = workflow.execute()
    assert result.success is True
```

**Running/Skipping Slow Tests:**
```bash
# Skip slow tests for fast local iteration
pytest -m "not slow"

# Run only slow tests
pytest -m slow

# Run all tests including slow (CI default)
pytest
```

#### Performance Tests (up to 5 minutes) - `@pytest.mark.performance`

Resource-intensive tests that benchmark performance, test under load, or validate behavior at scale. These are **excluded from standard CI/CD** and run separately (nightly, pre-release, or manually).

**Characteristics:**
- Benchmark tests measuring execution time
- Load tests with many iterations
- Stress tests with large data volumes
- Tests requiring significant compute resources

**Example:**
```python
@pytest.mark.performance
def test_large_file_processing_performance(benchmark):
    """Benchmark large file processing - may take minutes."""
    large_data = generate_test_data(size_mb=100)
    
    result = benchmark(process_large_file, large_data)
    
    # Assert performance baseline
    assert benchmark.stats["mean"] < 60.0  # Under 60 seconds
```

**Running Performance Tests:**
```bash
# Performance tests are excluded by default in CI
pytest -m "not performance"

# Run only performance tests (manual/nightly)
pytest -m performance

# Run performance benchmarks with detailed output
pytest -m performance --benchmark-only
```

### Configuring Test Markers

Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (>1s runtime) (deselect with '-m \"not slow\"')",
    "performance: marks tests as performance/benchmark (excluded from CI)",
    "integration: marks tests as integration tests",
    "e2e: marks tests as end-to-end tests",
]
```

Apply `slow` to tests consistently taking >1s (prefer class-level when all methods in a class are slow). Development flows:

```bash
# Fast feedback: skip slow tests
pytest adw/ -m "not slow" -q

# Exercise only slow tests (before CI/full runs)
pytest adw/ -m slow -q

# CI still runs the full suite by default
pytest adw/ -q
```

### CI/CD Configuration

Standard CI runs should exclude slow, performance, and integration tests, keep coverage enabled, and use xdist for in-runner parallelism:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pytest \
      -o addopts="-v --import-mode=importlib --cov=adw --cov-fail-under=80 --cov-report=term-missing -n auto --dist=loadscope" \
      -m "not slow and not performance and not integration"
```

Notes:
- Coverage is not split into a separate CI job; the standard pytest job already enforces coverage.
- Prefer `-n auto --dist=loadscope` on GitHub-hosted runners for balanced module-level parallelism.
- Emit `coverage.xml` from a single matrix leg when uploading to Codecov to avoid duplicate uploads.

Performance tests run in a separate job (nightly or manual):

```yaml
# .github/workflows/performance.yml (separate workflow)
- name: Run performance tests
  run: pytest -v -m performance --benchmark-json=benchmark.json
```

### Writing Tests with Duration in Mind

1. **Default to fast**: Write isolated, mocked tests unless integration is required
2. **Mark appropriately**: If a test takes > 1 second, consider if it needs `@pytest.mark.slow`
3. **Justify performance tests**: Only use `@pytest.mark.performance` for true benchmarks
4. **Keep CI fast**: Patch retry sleeps, use `pytest-xdist` in CI, and keep the standard suite under 5 minutes
5. **Document slow tests**: Add a comment explaining why the test is slow

## Platform Parity Testing

Platform parity tests ensure GitHub and GitLab implementations return the same
normalized structures across issues, labels, pull requests/merge requests,
comments, statuses, and workflow routing. Use the lightweight parity fixtures to
avoid network calls while keeping platform behavior aligned.

### Fixtures and Setup

- Parity clients (GitHub fork, GitLab upstream) live in
  `adw/platforms/tests/conftest.py` and share deterministic in-memory data.
- The router is rebuilt per test to prefer fork scope for writes and fall back
  to upstream when the fork lacks permissions.
- The mock GitLab server now exposes a minimal status endpoint so status parity
  checks use the same normalized payloads as GitHub.
- Router state is reset between tests (`_reset_platform_router`) to prevent
  leakage across modules.

### Running the Parity Suite

```bash
pytest adw/platforms/tests/workflow_parity_test.py -v
pytest adw/workflows/operations/tests/workflow_operations_parity_test.py -v
```

### Extending Parity Coverage

- Add new operations by extending the parity clients with minimal normalized
  helpers (prefer deterministic fixtures over large datasets).
- Keep datasets tiny (1–2 entities) to maintain speed while exercising router
  selection and status mocks.
- Mirror assertions across platforms (field-by-field comparisons, lenient
  timestamps) and include docstrings describing the methodology for each test.
- When adding new router behaviors, update the workflow parity tests to cover
  fork-preferred routing with upstream fallback and ensure error cases raise the
  same exception classes.

## Resolving Test Failures

### Test Failure Resolution Workflow

When tests fail, follow this systematic approach:

#### 1. Analyze the Test Failure

- Review the test name, purpose, and error message
- Understand what the test is trying to validate
- Identify the root cause from the error details

**Example Test Failure Input:**
```json
{
  "test_name": "test_cli_help",
  "passed": false,
  "execution_command": "uv run pytest",
  "test_purpose": "Verify CLI help command displays usage information",
  "error": "AssertionError: Expected exit code 0, got 1"
}
```

#### 2. Context Discovery

- Check recent changes: `git diff origin/main --stat --name-only`
- If a relevant spec exists, read it to understand requirements
- Focus only on files that could impact this specific test

#### 3. Reproduce the Failure

- Use the `execution_command` provided in the test data
- Run it to see the full error output and stack trace
- Confirm you can reproduce the exact failure

#### 4. Fix the Issue

- Make minimal, targeted changes to resolve only this test failure
- Ensure the fix aligns with the test purpose
- Do not modify unrelated code or tests

#### 5. Validate the Fix

- Re-run the same `execution_command` to confirm the test now passes
- Do NOT run other tests or the full test suite at this stage
- Focus only on fixing this specific test

### Key Principles for Test Failure Resolution

1. **Fix only the specific failing test** - Don't try to fix all tests at once
2. **Use the execution_command** - Always run the exact command that failed
3. **Minimal changes** - Make the smallest change that fixes the issue
4. **Understand the purpose** - Know what the test is validating before fixing
5. **Read specs** - If a spec exists, ensure your fix aligns with requirements

## Quick Reference

### Most Common Commands

```bash
# Run all tests (coverage included by default)
pytest

# Run specific test file
pytest adw/core/tests/agent_test.py

# Run tests with extra coverage artifacts
pytest --cov-report=html --cov-report=xml

# Run full validation sequence
ruff check adw/
ruff format --check adw/
mypy adw/ --ignore-missing-imports
pytest --cov-report=html --cov-report=xml
```

### File Naming Cheat Sheet

```
✓ CORRECT: agent_test.py                    # Tests for agent.py module
✓ CORRECT: workflow_operations_test.py      # Tests for workflow operations
✓ CORRECT: guide_references_test.py         # Integration test for guide references
✓ CORRECT: portability_compliance_test.py   # Integration test for portability

✗ WRONG: test_agent.py                      # Wrong prefix pattern
✗ WRONG: agent.test.py                      # Dot separator instead of underscore
✗ WRONG: agent_tests.py                     # Plural "tests" instead of "test"
```

## Examples

### Correct Test File Structure

#### Example 1: Simple Unit Test

```python
"""Tests for health check utilities."""

from adw.utils.health import run_health_check


class TestRunHealthCheck:
    """Tests for run_health_check function."""

    def test_health_check_runs_without_error(self, capsys):
        """Test that health check runs without raising exceptions."""
        # Should not raise any exceptions
        run_health_check()

        # Capture output
        captured = capsys.readouterr()

        # Should produce some output (Rich Panel)
        assert "System healthy" in captured.out or captured.out != ""

    def test_health_check_callable(self):
        """Test that health check function exists and is callable."""
        assert run_health_check is not None
        assert callable(run_health_check)
```

#### Example 2: Test with Mocking

```python
"""Tests for agent module."""

from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from adw.core.agent import check_opencode_installed


def test_check_opencode_installed_success():
    """Test when OpenCode CLI is installed."""
    mock_result = Mock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        result = check_opencode_installed()
    assert result is None


def test_check_opencode_installed_not_found():
    """Test when OpenCode CLI is not found."""
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        result = check_opencode_installed()
    assert result is not None
    assert "not installed" in result
```

#### Example 3: Parametrized Test

```python
"""Tests with multiple test cases."""

import pytest


@pytest.mark.parametrize("model_set,expected_model", [
    ("base", "sonnet"),
    ("heavy", "opus"),
])
def test_model_selection(model_set, expected_model):
    """Test model selection for different model sets."""
    # Test implementation
    assert True  # Replace with actual assertion
```

### Test Structure Best Practices

1. **Use descriptive test names**: Test function names should clearly describe what is being tested
2. **Use docstrings**: Every test should have a docstring explaining its purpose
3. **Group tests in classes**: Related tests should be grouped in test classes
4. **One assertion per test**: Focus each test on a single behavior (when possible)
5. **Use mocking**: Mock external dependencies to isolate the unit under test
6. **Use parametrize for variants**: When testing multiple inputs, use `@pytest.mark.parametrize`
7. **Clean up resources**: Use fixtures or context managers to clean up temporary files/resources

## Notebook Testing

Use the notebook tooling to catch corruption before execution and keep paired `.ipynb`/`.py`
files in sync.

- **Validate before execute:** Run `validate_notebook <path>` (or `--recursive` for trees) to
  check nbformat schema, required fields, and code cell syntax. Hidden files and
  `.ipynb_checkpoints/` are skipped automatically.
- **Auto-validation in execution:** `run_notebook` calls validation first; override only for
  debugging with `--skip-validation`. Execution overwrites the source by default and writes a
  `.ipynb.bak` backup unless you pass `--no-overwrite` or `--no-backup`.
- **Magics handling:** `%` line magics are stripped before syntax checks; `%%` cell magics are
  skipped so validation is not blocked.
- **Jupytext for lint/type-check:** Convert with `validate_notebook <path> --convert-to-py` and
  run `mypy`/`ruff` on the generated `.py` file. Use `--sync` to keep `.ipynb` and `.py`
  aligned; `--check-sync` fails CI when they diverge (no writes).
- **Exit codes:** `0` = valid/in-sync, `1` = validation or sync failure, `2` = tool/runtime error
  (e.g., dependency missing).
- **CI guidance:** Prefer `--output json` for machine-readable results. Run both validation
  and `--check-sync` in CI before executing notebooks. Add `run_notebook --skip-validation` only
  when validating the validator itself.

Example CI snippet:
```yaml
- name: Validate notebooks
  run: validate_notebook notebooks/ --recursive --output-mode json

- name: Enforce Jupytext sync
  run: validate_notebook notebooks/ --check-sync --recursive
```

## Troubleshooting

<!-- TODO: Add common issues specific to your testing setup -->

### Common Issues and Solutions

#### Issue 1: Tests Not Discovered

**Symptom:**
```
collected 0 items
```
or
```
No tests found
```

**Possible Causes and Solutions:**

1. **Wrong file naming**: Ensure test files match `*_test.py` suffix pattern
2. **Wrong directory**: Ensure tests are in correct location (module-level `tests/` subdirectory or `adw/tests/` for integration tests)
3. **Excluded by configuration**: Check test framework configuration for exclusions

#### Issue 2: Test Timeout

**Symptom**: Tests killed after 2 minutes.

**Cause**: Test execution exceeds timeout limit.

**Solutions:**
- Optimize slow tests
- Split large test suites
- Increase timeout if necessary
- Use test markers to categorize slow tests

#### Issue 3: Coverage Report Missing Modules

**Symptom**: Expected modules don't appear in coverage report.

**Cause**: Modules may be excluded by coverage configuration.

**Solution**: Review coverage configuration for exclusion patterns.

## Backend Testing

### Overview

The ADW backend abstraction layer (`adw/backends/`) provides a unified interface for multiple agent CLI backends (OpenCode). This section documents testing patterns specific to backend implementations.

### Test Categories

Backend tests are organized into four categories using pytest markers:

1. **Unit Tests** (no marker): Test individual backend methods in isolation
2. **Integration Tests** (`@pytest.mark.integration`): Test backend + workflow combinations
3. **End-to-End Tests** (`@pytest.mark.e2e`): Test complete workflow scenarios
4. **Performance Tests** (`@pytest.mark.performance`): Benchmark backend operations

### Running Backend Tests

```bash
# Run all backend tests
pytest adw/backends/tests/ -v

# Run only unit tests (exclude integration, e2e, performance)
pytest adw/backends/tests/ -v -m "not integration and not e2e and not performance"

# Run integration tests
pytest adw/backends/tests/ -v -m integration

# Run end-to-end tests
pytest adw/backends/tests/ -v -m e2e

# Run performance benchmarks
pytest adw/backends/tests/ -v -m performance

# Run with coverage
pytest adw/backends/tests/ -v --cov=adw.backends --cov-report=term-missing
```

### Test File Organization

Backend tests follow the flat structure pattern with descriptive file names:

```
adw/backends/tests/
├── conftest.py                              # Shared fixtures
├── base_test.py                             # Unit tests for abstract base
├── claude_test.py                           # Unit tests for Claude backend
├── opencode_test.py                         # Unit tests for OpenCode backend
├── factory_test.py                          # Unit tests for factory
├── config_test.py                           # Unit tests for configuration
├── models_test.py                           # Unit tests for data models
├── workflow_claude_integration_test.py      # Integration tests: Claude + workflows
├── workflow_opencode_integration_test.py    # Integration tests: OpenCode + workflows
├── backend_switching_integration_test.py    # Integration tests: backend switching
├── complete_workflow_e2e_test.py            # E2E tests: complete workflows
├── error_scenarios_e2e_test.py              # E2E tests: error handling
├── benchmark_performance_test.py            # Performance benchmarks
└── performance_baseline.json                # Performance baselines (data file)
```

### Mocking Patterns

Backend tests extensively mock external dependencies to ensure fast, isolated tests.

#### Mocking time.sleep for Fast Tests

Tests involving retry decorators or backoff delays should mock `time.sleep` to avoid slow test execution. When mocking sleep, the mock parameter appears **first** in the test method signature (decorators apply bottom-to-top):

```python
from unittest.mock import patch

@pytest.mark.slow
@patch("adw.platforms.decorators.time.sleep", autospec=True)
@patch("adw.github.client.Github")
@patch("adw.github.client.GithubIntegration")
@patch("adw.github.client.Auth")
def test_get_repo_github_exception(
    self, mock_auth, mock_gi_class, mock_github_class, mock_sleep
):
    """Test retry behavior when GitHub raises an exception.
    
    Note: mock_sleep is last in decorator order but first in parameter list
    because decorators are applied bottom-to-top.
    """
    mock_github_class.return_value.get_repo.side_effect = GithubException(500, "Error")
    
    with pytest.raises(GithubException):
        get_repo("owner/repo")
    
    # Verify sleep was called during retries (confirms fast execution)
    assert mock_sleep.called
```

**Key points:**
- Use `autospec=True` to ensure the mock matches the real `time.sleep` signature
- Mark tests with `@pytest.mark.slow` if they would be slow without the mock
- Mock the sleep in the module where the decorator is defined (e.g., `adw.platforms.decorators.time.sleep`)
- Decorator order matters: the first `@patch` decorator corresponds to the **last** parameter

#### Mocking CLI Execution

```python
from unittest.mock import patch

@patch("adw.backends.opencode.execute_agent")
def test_backend_execution(mock_prompt, mock_success_response):
    """Test backend executes prompts successfully."""
    # Setup mock
    mock_prompt.return_value = mock_success_response

    # Execute test
    backend = OpenCodeBackend()
    response = backend.execute_prompt(request)

    # Verify
    assert response.success is True
```

#### Mocking Environment Variables

```python
def test_config_from_env(monkeypatch):
    """Test configuration loads from environment."""
    monkeypatch.setenv("ADW_BACKEND", "opencode")
    monkeypatch.setenv("ADW_OPENCODE_CLI_PATH", "/usr/bin/opencode")

    # Test configuration loading
    config = BackendConfigManager().load_backend_config()
    assert config.cli_path == "/usr/bin/opencode"
```

#### Mocking Subprocess Calls

```python
from unittest.mock import Mock

@patch("subprocess.run")
def test_version_check(mock_run):
    """Test CLI version retrieval."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "opencode 1.0.0"
    mock_run.return_value = mock_result

    backend = OpenCodeBackend()
    version = backend.get_version()
    assert version == "1.0.0"
```

### Using Shared Fixtures

The `conftest.py` file provides reusable fixtures for backend testing:

```python
def test_with_fixtures(opencode_backend, mock_success_response):
    """Test using shared fixtures."""
    # opencode_backend fixture provides OpenCodeBackend instance
    # mock_success_response provides AgentPromptResponse

    assert opencode_backend.cli_path == "opencode"
    assert mock_success_response.success is True
```

Common fixtures include:
- `opencode_backend`: Backend instances
- `mock_success_response` / `mock_error_response` / `mock_timeout_response`: Response fixtures
- `backend_config_opencode`: Configuration fixtures
- `small_prompt` / `medium_prompt` / `large_prompt`: Test prompt fixtures
- `mock_env_opencode`: Environment fixtures

### Writing Integration Tests

Integration tests validate backend + workflow combinations:

```python
@pytest.mark.integration
class TestOpenCodeBackendPlanWorkflow:
    """Integration tests for plan workflow with OpenCode backend."""

    @patch("adw.backends.opencode.execute_agent")
    def test_plan_workflow_success(self, mock_prompt, opencode_backend, mock_success_response):
        """Test plan workflow executes successfully."""
        mock_prompt.return_value = mock_success_response

        request = AgentTemplateRequest(
            slash_command="/plan",
            args=["issue.json"],
            adw_id="test123",
            agent_name="planner",
            model="sonnet",
        )

        response = opencode_backend.execute_template(request)
        assert response.success is True
```

### Writing End-to-End Tests

E2E tests validate complete workflow scenarios:

```python
@pytest.mark.e2e
class TestCompleteWorkflowOpenCode:
    """End-to-end tests for complete workflow."""

    @patch("adw.backends.opencode.execute_agent")
    def test_complete_workflow(self, mock_prompt, mock_success_response):
        """Test complete workflow: plan → build → test → review."""
        mock_prompt.return_value = mock_success_response

        factory = BackendFactory()
        backend = factory.get_backend("opencode")

        # Execute workflow steps
        for slash_command, args in [("/plan", ["issue.json"]),
                                     ("/implement", ["plan.md"]),
                                     ("/test", []),
                                     ("/review", [])]:
            request = AgentTemplateRequest(
                slash_command=slash_command,
                args=args,
                adw_id="test123",
                agent_name="agent",
                model="sonnet",
            )
            response = backend.execute_template(request)
            assert response.success is True
```

### Writing Performance Tests

Performance tests benchmark key operations:

```python
@pytest.mark.performance
class TestBackendPerformance:
    """Performance benchmarks for backend operations."""

    def test_backend_init_performance(self, benchmark):
        """Benchmark backend initialization time."""
        def init_backend():
            return OpenCodeBackend()

        backend = benchmark(init_backend)
        assert backend is not None
```

### Performance Baselines

Performance baselines are documented in `performance_baseline.json`. Key baselines:

- Backend initialization: < 100ms
- Configuration loading: < 50ms
- Factory singleton cache: < 1ms
- Template execution (mocked): < 3s

These baselines measure framework overhead only (CLI execution is mocked).

### Testing Both Backends

Always test the OpenCode backend:

```python
@pytest.mark.parametrize("backend_type,model", [
    ("opencode", "sonnet"),
])
def test_backend(backend_type, model):
    """Test feature works with backend."""
    factory = BackendFactory()
    backend = factory.get_backend(backend_type)

    # Test implementation
    assert backend is not None
```

### Common Testing Scenarios

#### Error Handling

```python
def test_cli_not_found(mock_prompt):
    """Test handling when CLI is not installed."""
    mock_prompt.return_value = AgentPromptResponse(
        output="Error: CLI not found",
        success=False,
        retry_code=RetryCode.FATAL_ERROR,
    )

    response = backend.execute_prompt(request)
    assert response.success is False
```

#### Timeout Handling

```python
def test_timeout_error(mock_timeout_response):
    """Test handling of timeout errors."""
    # Use timeout response fixture
    assert mock_timeout_response.retry_code == RetryCode.RECOVERABLE_ERROR
```

#### Configuration Variations

```python
def test_custom_cli_path():
    """Test backend with custom CLI path."""
    backend = OpenCodeBackend(cli_path="/custom/path/opencode")
    assert backend.cli_path == "/custom/path/opencode"
```

### Coverage Goals

Backend tests target **90% coverage** of the `adw.backends` package:

```bash
pytest adw/backends/tests/ --cov=adw.backends --cov-report=term-missing --cov-fail-under=90
```

Focus coverage on:
- All public methods of backend classes
- Configuration loading and validation
- Error handling paths
- Response parsing and normalization

### Troubleshooting Backend Tests

#### Issue: Tests fail with "CLI not found"

**Solution**: Ensure CLI execution is mocked. Backend tests should never execute real CLI commands.

```python
# Always mock CLI execution
@patch("adw.backends.opencode.execute_agent")
def test_backend(mock_prompt):
    # Test implementation
```

#### Issue: Tests are slow

**Solution**: Verify all external dependencies are mocked. Backend unit tests should complete in milliseconds.

#### Issue: Integration tests fail

**Solution**: Check that workflow dependencies (GitHub, Git, file system) are properly mocked.

## GitLab Integration Testing

ADW provides comprehensive mock infrastructure for testing GitLab platform operations
without requiring real GitLab credentials. This enables CI testing and local development
against GitLab's REST API v4 endpoints.

### MockGitLabServer

The `MockGitLabServer` class simulates GitLab API responses using the `responses` library
to intercept HTTP calls made by `python-gitlab`. It supports:

- **Issues**: List, get, create, and comment operations
- **Merge Requests**: List and create operations
- **Labels**: List and create operations
- **Error Simulation**: 404, 401, and 429 rate limit responses

#### Basic Usage

```python
from adw.platforms.tests.mock_gitlab_server import MockGitLabServer

# As a context manager (recommended)
with MockGitLabServer() as server:
    server.add_issue(1, title="Test Issue", state="opened")
    server.add_label("bug", "#FF0000")
    # Run tests against mock server

# Manual start/stop
server = MockGitLabServer(
    base_url="https://gitlab.com",
    project_id="owner/repo",
)
server.start()
# ... run tests ...
server.stop()
```

#### Error Simulation

```python
with MockGitLabServer() as server:
    # Simulate rate limit (429 responses)
    server.simulate_rate_limit()
    # All API calls now return 429

    # Simulate auth failure (401 responses)
    server.simulate_auth_failure()
    # All API calls now return 401

    # Reset to normal operation
    server.reset_error_simulation()
```

### Running GitLab Integration Tests

```bash
# Run all GitLab integration tests
pytest adw/platforms/tests/gitlab_integration_test.py -v

# Run specific test class
pytest adw/platforms/tests/gitlab_integration_test.py::TestGitLabIntegration -v
pytest adw/platforms/tests/gitlab_integration_test.py::TestCrossPlatformIntegration -v

# Run with coverage
pytest adw/platforms/tests/gitlab_integration_test.py -v --cov=adw.platforms
```

### Test Categories

The GitLab integration tests are organized into three classes:

| Test Class | Purpose | Test Count |
|------------|---------|------------|
| `TestGitLabIntegration` | Core GitLab operations (issues, MRs, labels, errors) | 8 |
| `TestCrossPlatformIntegration` | Platform detection, router configuration, dual-platform | 5 |
| `TestGitLabWorkflowDispatch` | Workflow dispatch (pending E1 integration) | 1 (skipped) |

### Shared Fixtures

The `adw/platforms/tests/conftest.py` module provides reusable fixtures:

```python
# mock_gitlab: Basic MockGitLabServer instance
def test_with_mock_gitlab(mock_gitlab):
    mock_gitlab.add_issue(1, title="Test Issue")
    # Test code using mock server

# reset_router_fixture: Auto-use fixture that resets the platform router
# singleton between tests (runs automatically for all platform tests)
```

### Extending the Mock Server

To add new endpoints to `MockGitLabServer`:

1. **Add a callback method**:
   ```python
   def _handle_get_branches(self, request):
       branches = [{"name": "main"}, {"name": "develop"}]
       return (200, {}, json.dumps(branches))
   ```

2. **Register the endpoint** in `_register_endpoints()`:
   ```python
   self._responses_mock.add_callback(
       responses.GET,
       re.compile(rf"{api_base}/repository/branches$"),
       callback=self._handle_get_branches,
   )
   ```

3. **Add data manipulation methods** if needed:
   ```python
   def add_branch(self, name: str) -> dict[str, Any]:
       branch = {"name": name, "protected": False}
       self.data.branches.append(branch)
       return branch
   ```

See `adw/platforms/tests/mock_gitlab_server.py` for the complete implementation
with detailed docstrings and usage examples.

## See Also

### Configuration Files

- **pyproject.toml**: Test framework configuration (`[tool.pytest.ini_options]`), coverage configuration (`[tool.coverage]`), and ruff linting configuration
- **.github/workflows/test.yml**: CI test workflow configuration
- **.github/workflows/lint.yml**: CI linting workflow configuration

### Related Documentation

- **linting_guide.md**: Linting conventions (affects test file linting)
- **architecture/decisions/ADR-002-backend-abstraction-layer.md**: Backend architecture
- **architecture/decisions/ADR-003-opencode-backend-implementation.md**: OpenCode implementation
- **README.md**: Project overview and setup instructions
- **CONTRIBUTING.md**: Contribution guidelines including testing requirements

### External Resources

- **pytest Documentation**: https://docs.pytest.org/
- **pytest-cov Documentation**: https://pytest-cov.readthedocs.io/
- **pytest-benchmark Documentation**: https://pytest-benchmark.readthedocs.io/
- **ruff Documentation**: https://docs.astral.sh/ruff/
- **mypy Documentation**: https://mypy.readthedocs.io/

---

**Questions or issues?** Consult project documentation or open an issue.
