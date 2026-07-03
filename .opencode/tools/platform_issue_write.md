# platform_issue_write

Wrapper for `adw platform create-issue` and `update-issue`.

## Supported commands

- `create-issue`
- `update-issue`

## Required arguments

- `command`: required command name
- `create-issue`: non-empty `title`
- `update-issue`: `issue_number` plus at least one of `title`, `body`, `labels`, or `state`

## Optional arguments

- `body`
- `labels`
- `state`: `open|closed` (`update-issue` only)
- `output_format`: `text|json`
- `prefer_scope`: `fork|upstream`
- `help`: bypasses required-argument validation and delegates `--help`

## Notes

- Issue identifiers must be positive integer tokens; all-zero values are rejected.
- Blank optional strings are omitted from command assembly.
- Failure diagnostics prefer `STDERR` then `STDOUT`; JSON-mode stdout stays sanitized and bounded.

## Examples

```json
{ "command": "create-issue", "title": "Bug: login fails", "body": "Details...", "labels": "bug" }
```

```json
{ "command": "update-issue", "issue_number": "123", "state": "closed", "output_format": "json" }
```
