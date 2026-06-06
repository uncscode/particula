# Pytest Runner Tool Reference

Full parameter reference for the run_pytest tool. For quick usage, see the tool description.

## Simple Examples

```jsonc
// Full test suite
{ "minTests": 1700 }

// Scoped tests (always set minTests: 1)
{ "pytestArgs": ["adw/core/tests/"], "minTests": 1 }

// Test by name
{ "pytestArgs": ["-k", "test_agent"], "minTests": 1 }

// Skip slow tests
{ "pytestArgs": ["-m", "not slow and not performance"], "minTests": 1 }

// Fail fast
{ "failFast": true, "pytestArgs": ["adw/core/tests/"], "minTests": 1 }

// With coverage threshold
{ "coverage": true, "coverageThreshold": 80 }

// In a worktree
{ "cwd": "/path/to/worktree", "pytestArgs": ["adw/"], "minTests": 1 }
```

## Advanced Examples

```jsonc
// Custom coverage source
{ "coverageSource": "adw.core,adw.utils" }

// Use pyproject.toml config (default behavior)
{}

// Show slowest 10 tests
{ "durations": 10, "minTests": 1 }

// Show all test durations
{ "durations": 0, "minTests": 1 }

// JSON output with durations
{ "durations": 10, "outputMode": "json", "minTests": 1 }

// Multiple coverage reports
{ "covReport": ["term-missing", "html:coverage_html"], "minTests": 1 }

// Skip coverage for speed
{ "coverage": false, "pytestArgs": ["adw/core/tests/"], "minTests": 1 }

// Override ini options
{ "overrideIni": ["addopts="], "minTests": 1 }
```

## Parameter Reference

| Parameter         | Type     | Default          | Description                              |
|-------------------|----------|------------------|------------------------------------------|
| `outputMode`      | enum     | "summary"        | Output: summary, full, json              |
| `minTests`        | number   | 1                | Minimum expected test count              |
| `pytestArgs`      | string[] | []               | Additional pytest arguments              |
| `timeout`         | number   | 600              | Timeout in seconds                       |
| `coverage`        | boolean  | true             | Enable coverage reporting                |
| `coverageSource`  | string   | pyproject.toml   | Module(s) for coverage (comma-separated) |
| `coverageThreshold` | number | —               | Minimum coverage % (0-100)               |
| `cwd`             | string   | project root     | Working directory (use for worktrees)    |
| `failFast`        | boolean  | false            | Stop on first failure (-x)               |
| `covReport`       | string[] | ["term-missing"] | Coverage report format(s)                |
| `durations`       | number   | —                | Show N slowest tests (0 = all)           |
| `durationsMin`    | number   | 0.005            | Min duration for slowest list            |
| `overrideIni`     | string[] | []               | Override pytest ini options               |

## Key Behaviors

- `-v` and `--tb=short` are always included -- do NOT pass these in pytestArgs.
- `minTests` validates the test count: set to 1 for scoped tests, ~1700 for full suite.
- When `coverageSource` is omitted or `"all"`, uses pyproject.toml `[tool.coverage.run].source`.
- Blank optional strings are treated as omitted.
- Positive numeric values required when provided (timeout, minTests).

## Output Modes

| Mode      | Description                              |
|-----------|------------------------------------------|
| `summary` | Human-readable summary only (default)    |
| `full`    | Complete pytest output + summary         |
| `json`    | Structured data for programmatic use     |
