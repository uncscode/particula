# run_pytest_basic

Routine pytest wrapper for non-advanced usage.

## Preferred wrapper

- Use `run_pytest_basic` for standard test execution and simple path targeting.

## Compatibility status

- This is the preferred split wrapper for routine pytest runs.
- It rejects advanced controls by key presence and returns a deterministic error directing callers to `run_pytest_advanced`.

## Direct fields

- `minTests`
- `timeout`
- `cwd`
- `testPath`

Keep routine targeting and path fields explicit.

## Bounded `options` tokens

- `output=<summary|full|json>`
- `fail-fast`
- `test-filter=<value>`

## Examples

```json
{ "testPath": "pkg/tests/agent_test.py", "minTests": 1 }
{ "options": "test-filter=agent", "minTests": 1 }
{ "options": "output=json fail-fast", "testPath": "pkg/tests/", "minTests": 1 }
{ "cwd": "/path/to/worktree", "testPath": "tests/guide_references_test.py", "minTests": 1 }
```

## Notes

- `timeout` is measured in seconds and must be greater than 0 and less than or equal to 3600 seconds (1 hour).
- `cwd` must resolve within the current repository root.
- Removed legacy direct fields (`outputMode`, `failFast`, `testFilter`) now fail closed and must move through `options`.
- Use `run_pytest_advanced` for coverage, durations, `overrideIni`, or passthrough `pytestArgs`.
