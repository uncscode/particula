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

- Missing `ast-grep` runtime/binary cases return `classification: missing_binary`
  plus an install/PATH hint.
- Invalid pattern/rewrite parse cases return `classification: parse_input`
  and tell the caller to fix the AST input rather than install tooling.
- Diagnostic precedence stays deterministic: `stderr`, then `stdout`, then the
  thrown message.
- This wrapper remains read-only preview mode and never appends `--update-all`.
