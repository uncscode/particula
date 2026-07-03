# run_pytest_advanced

Advanced pytest wrapper for coverage and reporting controls.

## Preferred wrapper

- Use `run_pytest_advanced` when you need coverage controls, durations, `overrideIni`, or passthrough `pytestArgs`.

## Compatibility status

- This is the preferred split wrapper for advanced pytest runs.
- Use `run_pytest_basic` instead for routine execution.

## Direct fields

- `minTests`
- `timeout`
- `cwd`
- `testPath`
- `pytestArgs`
- `coverage`
- `coverageSource`
- `coverageThreshold`
- `overrideIni`

Keep advanced payload-bearing fields explicit.

## Bounded `options` tokens

- `output=<summary|full|json>`
- `fail-fast`
- `test-filter=<value>`
- `cov-report=<csv>`
- `durations=<n>`
- `durations-min=<n>`

## Examples

```json
{ "pytestArgs": ["pkg/tests/"], "minTests": 1 }
{ "coverage": true, "coverageThreshold": 80, "minTests": 1 }
{ "options": "output=json durations=10", "pytestArgs": ["tests/"], "minTests": 1 }
{ "options": "test-filter=agent fail-fast", "pytestArgs": ["tests/"], "minTests": 1 }
{ "overrideIni": ["addopts="], "minTests": 1 }
```

## Notes

- `timeout` is measured in seconds and must be greater than 0 and less than or equal to 3600 seconds (1 hour).
- `coverage: false` emits the no-coverage path.
- `durations=0` is supported and means show all durations.
- `durations-min=<n>` only takes effect when `durations=<n>` is also set.
- `cwd` must resolve within the current repository root.
- `coverageSource` accepts module/package names, repo-relative directories, and repo-relative file targets. Empty comma-separated segments, absolute paths, and repo/worktree-escaping traversal paths are rejected before spawn.
- Repo-relative file-target coverage requests may succeed with `"coverage_files": null` when per-file numeric detail is intentionally non-authoritative.
- Coverage-enabled runs in the same worktree are serialized. If another coverage run already holds the worktree lock, the wrapper fails deterministically instead of sharing `.coverage` artifacts.
- Raw coverage-related `pytestArgs` are rejected when `coverage` is `false`.
- Removed legacy direct fields (`outputMode`, `failFast`, `testFilter`, `covReport`, `durations`, `durationsMin`) now fail closed and must move through `options`.
