# ADW Issues Batch Init Tool Reference

Initializes issue-batch state through the issue-batch CLI backend.

Prefer the split wrapper `adw_issues_batch_init` for all active usage. The
legacy compatibility wrapper `adw_issues_spec` has been retired and archived
under `.trash/`.

## Supported Command

| Command | Required params | Description |
|---|---|---|
| `batch-init` | `total`, `source` | Initialize batch content from a source document |

## Examples

```jsonc
// Preferred split-wrapper usage
{ "total": "5", "source": ".opencode/plans/sections/features/F42/issue.md" }

// Resume or target a specific workflow state file
{ "total": "5", "source": "docs/feature.md", "adw_id": "abc12345" }

// Historical retired-wrapper equivalent
{ "command": "batch-init", "total": "5", "source": "docs/feature.md", "adw_id": "abc12345" }
```

## Parameter Reference

| Parameter | Type | Required | Notes |
|---|---|---|---|
| `total` | string | Yes | Positive integer between 1 and 50 |
| `source` | string | Yes | Safe non-empty relative path |
| `adw_id` | string | No | Optional 8-character hex workflow ID |
| `options` | string | No | No command-scoped option tokens are currently supported |

## Notes

- `source` must stay relative and must not include traversal segments.
- `options` is accepted for parity with the compatibility surface, but
  `batch-init` currently has no bounded option tokens.
