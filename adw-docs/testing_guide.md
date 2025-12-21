# Testing Guide

This guide documents testing conventions for {{PACKAGE_NAME}}. It favors fast, isolated unit tests and uses pytest for all suites.

## Directory layout

- Place tests next to the code they cover in module-level `tests/` directories.
- Keep helpers/fixtures in `conftest.py` per directory to share setup without import cycles.
- Avoid importing application modules inside `conftest.py` top-level scope if it triggers expensive side effects; prefer fixtures.

Example structure:
```
{{PACKAGE_NAME}}/
  module_a/
    __init__.py
    service.py
    tests/
      service_test.py
  module_b/
    __init__.py
    handler.py
    tests/
      handler_test.py
```

## File naming

All test files must follow the `{{TEST_FILE_PATTERN}}` suffix pattern.

**Correct:**
```
agent_test.py
workflow_test.py
```

**Wrong:**
```
test_agent.py
agent_tests.py
```

Keep test function names descriptive: `test_handles_invalid_payload` is better than `test_error`.

## Writing tests

- Prefer **pure functions** and **small fixtures**; use factories/builders for complex objects.
- Use markers for expensive tests: `@pytest.mark.slow`, `@pytest.mark.performance`, `@pytest.mark.integration`.
- Keep fast tests (<1s) in the default suite; gate slow/integration tests behind markers and only run them explicitly.
- Assert on behavior and side effects, not implementation details.
- When mocking, patch at the boundary (e.g., HTTP clients, filesystem, time). Avoid over-mocking core logic.

## Running tests

```bash
# Default fast suite
{{TEST_COMMAND}}

# With coverage (recommended for CI)
{{TEST_COMMAND}} --cov={{PACKAGE_NAME}} --cov-report=term-missing

# Focused module or test
{{TEST_COMMAND}} {{PACKAGE_NAME}}/module_a/tests/service_test.py -k "happy_path"
```

Coverage target: **{{COVERAGE_THRESHOLD}}%** minimum for new/changed code.

## Markers and selection

- **Skip slow/performance** in fast runs: `{{TEST_COMMAND}} -m "not slow and not performance"`
- **Run integration tests** when needed: `{{TEST_COMMAND}} -m integration`
- **Fail fast** during debugging: `{{TEST_COMMAND}} -x --maxfail=1`

## Fixtures and data

- Put shared fixtures in the nearest `conftest.py` to limit scope.
- Use `tmp_path`/`tmp_path_factory` for filesystem isolation.
- Prefer deterministic sample data; randomize only with seeded values.

## Import and style alignment

- Follow the import order and typing rules in [Code Style](./code_style.md).
- Keep test files formatted with the same line length ({{LINE_LENGTH}}) and linted via `{{LINT_COMMAND}}` to avoid drift.

## Troubleshooting

- If tests fail unexpectedly, run with `-vv --maxfail=1` to see detailed assertion diffs.
- A snapshot of failing cases helps reviewers—paste key traceback lines in PRs.
- For flaky tests, identify external dependencies (time, network, randomness) and remove them or mark as flaky until fixed.

## Related guides

- [Code Style](./code_style.md)
- [Linting Guide](./linting_guide.md)
