# refactor_astgrep_preview

Non-mutating ast-grep wrapper.

- Runs: `ast-grep run -p <pattern> -r <rewrite> -l <lang> -- <path>`
- Never appends `--update-all`
- Returns preview output or `No matches found for pattern: <pattern>`

## Example

```json
{ "pattern": "old_name($$$ARGS)", "rewrite": "new_name($$$ARGS)", "lang": "python", "path": "adw" }
```

## Errors

Deterministic `ERROR:` envelopes are returned for execution failures.
