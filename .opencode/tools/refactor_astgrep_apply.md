# refactor_astgrep_apply

Mutating ast-grep wrapper.

- Runs base command then appends `--update-all`
- Returns command output or `No files modified (no matches).`

## Example

```json
{ "pattern": "old_name($$$ARGS)", "rewrite": "new_name($$$ARGS)", "lang": "python", "path": "adw" }
```

## Partial-failure recovery

Apply mode is not transactional. If ast-grep fails mid-run, inspect and recover manually:

1. `git diff` to inspect partial edits
2. `git restore <paths>` (or equivalent)
3. Fix the pattern/rewrite and retry

Execution failures return deterministic `ERROR:` envelopes with remediation hints.
