# run_pytest_advanced

Advanced pytest wrapper for coverage and reporting controls.

For `output=json`, every runner-owned result includes the exact top-level
`evidence_identity` object `{"contract":"e37-m2-validation-git","version":1}`.
This includes completed, collection, validation/pre-launch, timeout, lease, and
unexpected-failure JSON paths; summary and full text output remain identity-free.
The TypeScript wrapper transports valid runner JSON unchanged. Its own fallback
envelope for invalid runner JSON is identity-free and is not producer evidence.
The identity is a bounded compatibility marker, not a success result or authority
to execute tests, mutate state, delegate, invoke a shell, use the network, or
perform lifecycle actions.

## Preferred wrapper

- Use `run_pytest_advanced` when you need coverage controls, durations, restricted `overrideIni`, or ordered `pytestArgs`.

## Compatibility status

- This is the preferred split wrapper for advanced pytest runs.
- Use `run_pytest_basic` instead for routine execution.

## Direct fields

- `minTests`
- `timeout`
- `cwd`
- `testPath`
- `testPaths`
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
{ "overrideIni": ["filterwarnings=error"], "minTests": 1 }
```

## Notes

- `timeout` is measured in seconds and must be greater than 0 and less than or equal to 1200 seconds (20 minutes).
- `coverage: false` emits the no-coverage path and rejects `coverageSource`,
  `coverageThreshold`, and `cov-report` controls before subprocess launch.
- `durations=0` is supported and means show all durations.
- `durations-min=<n>` only takes effect when `durations=<n>` is also set.
- `cwd` must resolve within the current repository root.
- JSON results for executed non-collection pytest runs include independent `assertion`
  (`passed|failed`) and `coverage` (`disabled|passed|failed`) projections, each
  with bounded `reasons`. Pre-spawn validation failures return the bounded
  `outcome` payload without these execution projections. Coverage includes
  `percentage` only when a numeric `TOTAL` value was reported.
  `disabled` is not a coverage pass; overall success requires passed assertions
  and either disabled or passed coverage.
- `coverageSource` accepts identifier-style package/module names or canonical,
  root-confined POSIX directories and existing regular `.py` files. `all` is
  case-insensitive but must be the only source. Empty segments, absolute paths,
  backslashes, traversal, noncanonical paths, unsafe/missing filesystem targets,
  and unsupported suffixes are rejected before spawn.
- Explicit sources are forwarded as repeated `--cov` controls. They preserve the
  repository coverage configuration; the runner does not create a temporary
  coverage configuration file.
- Coverage requires usable numeric `TOTAL` evidence. Missing totals and
  no-data/never-imported diagnostics fail coverage independently. The repository
  80% floor is retained; a caller threshold can strengthen it but cannot lower it.
- Repo-relative file-target coverage requests may succeed with `"coverage_files": null` when per-file numeric detail is intentionally non-authoritative.
- Coverage-enabled runs in the same worktree are serialized. If another coverage run already holds the worktree lock, the wrapper fails deterministically instead of sharing `.coverage` artifacts.
- Coverage coordination uses a canonical-worktree, no-wait ownership lease under
  `adforge_local/state/`. A verified live lease returns bounded retry guidance;
  publication is atomic, state paths reject symlinks, and stale recovery and
  release are serialized with ownership-safe transitions.
  It never queues, polls, or exposes lease paths, holder identifiers, or tokens.
- `pytestArgs` is a literal ordered array. Accepted entries are transported once as a compact JSON
  array; they are never trimmed, split, joined, or appended as runner arguments.
- The permitted caller grammar is confined path/node-id targets, `-k VALUE`, `-m VALUE`,
  `-p VALUE`, restricted non-`addopts` `--override-ini=VALUE`, `--collect-only`, `-q`, `-v`,
  `--verbose`, and `--tb=short|long|line|native|no`. Caller `-o` and `addopts` controls are rejected.
- Raw coverage and runner controls (output, timeout, cwd, targeting, fail-fast, durations, and
  transport controls) are prohibited in `pytestArgs`; use their dedicated fields/tokens instead.
- `testPath` and `test-filter` use runner-owned named transport. `testPath` stays repository-relative
  even when `cwd` is nested; the runner converts it safely for execution. `overrideIni` is independently
   transported as an array, and caller `addopts` controls are rejected.
- `testPaths` accepts one through seven ordered canonical, repository-confined POSIX
  path/node-id targets and transports them once as compact `--test-paths-json`.
  It cannot be combined with `testPath`, including empty-field ambiguity.
- `--collect-only` in `pytestArgs` requires explicit `coverage: false`. Collection
  reports collected and zero-executed counts, applies `minTests` to collected
  tests, and intentionally produces no assertion or coverage evidence. A
  successful JSON collection result includes `success: true`, `collection`, and
  `outcome: {"classification":"collection","status":"completed"}`.
- Failed JSON output, including timeout and pre-spawn failures, includes one bounded `outcome` with classification, reason, exit/phase,
  resolved target, node IDs, output tails, and per-field truncation counts. It never exposes the
  raw command, absolute cwd, or uncapped subprocess output.
- Removed legacy direct fields (`outputMode`, `failFast`, `testFilter`, `covReport`, `durations`, `durationsMin`) now fail closed and must move through `options`.
