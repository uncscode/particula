# platform_pr_write

Wrapper for `adw platform create-pr` with compatibility success/failure markers.

## Supported commands

- `create-pr`

## Required arguments

- `command`: required command name
- `title`: non-empty title
- `head`: non-empty source branch

## Optional arguments

- `body`
- `base`
- `adw_id`: 8-character lowercase hex string
- `draft`: boolean
- `prefer_scope`: `fork|upstream`
- `help`: bypasses required-argument validation and delegates `--help`

## Notes

- Success returns `PLATFORM_PR_CREATED` compatibility output.
- Failures return `PLATFORM_PR_FAILED` with sanitized diagnostics.
- Blank optional strings are omitted from command assembly.

## Example

```json
{ "command": "create-pr", "title": "feat: #123 - Add auth", "head": "feature-123", "adw_id": "abc12345" }
```
