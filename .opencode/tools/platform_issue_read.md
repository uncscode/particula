# platform_issue_read

Read-only wrapper for `adw platform fetch-issue`.

## Supported commands

- `fetch-issue`

## Arguments

- `command`: required command name
- `issue_number`: required positive integer token
- `output_format`: optional `text` or `json`
- `prefer_scope`: optional `fork` or `upstream`
- `help`: show delegated CLI help

## Notes

- Required identifiers are validated before subprocess execution.
- JSON-mode failure output stays sanitized and bounded.
- Success output is delegated directly from `uv run --active adw platform fetch-issue`.

## Example

```json
{ "command": "fetch-issue", "issue_number": "123", "output_format": "json" }
```
