# refactor_astgrep_preview

Non-mutating ast-grep wrapper.

- Runs: `ast-grep run -p <pattern> -r <rewrite> -l <lang> -- <path>`
- Never appends `--update-all`
- Returns a bounded typed object with exactly one of `match`, `no_match`,
  `parse_error`, `unavailable`, or `runtime_error` in its `kind` field.

## Example

```json
{ "pattern": "old_name($$$ARGS)", "rewrite": "new_name($$$ARGS)", "lang": "python", "path": "adw" }
```

## Errors

Preview never returns an `ERROR:` envelope. Failure variants contain a sanitized
diagnostic capped at 500 characters and an optional equally bounded hint.

- Missing `ast-grep` runtime/binary cases return `kind: "unavailable"` plus an
  install/PATH hint.
- Invalid pattern/rewrite parse cases return `kind: "parse_error"` and tell the
  caller to fix the AST input rather than install tooling.
- Diagnostic precedence stays deterministic: `stderr`, then `stdout`, then the
  thrown message.
- This wrapper remains read-only preview mode and never appends `--update-all`.
- `match` previews are capped at 16,000 characters and 200 nonempty lines.
- Target paths are lexically and canonically confined to the repository before execution.
