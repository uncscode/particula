# platform_pr_read

Read-only wrapper for pull-request / merge-request read commands.

## Supported commands

- `pr-comments`
- `pr-diff`

## Arguments

- `command`: required command name
- `issue_number`: required PR/MR number token
- `output_format`: optional `text` or `json`
- `prefer_scope`: optional `fork` or `upstream`
- `actionable_only`: supported only for `pr-comments`
- `help`: show delegated CLI help

## Examples

```json
{ "command": "pr-comments", "issue_number": "42", "output_format": "json" }
```

```json
{ "command": "pr-comments", "issue_number": "42", "actionable_only": true, "prefer_scope": "fork" }
```

```json
{ "command": "pr-diff", "issue_number": "42", "output_format": "json", "prefer_scope": "upstream" }
```

## Notes

- Required PR/MR identifiers are validated before subprocess execution.
- `pr-diff` rejects `actionable_only` during wrapper preflight.
- Success output is delegated directly from `uv run adw platform <command>`.
