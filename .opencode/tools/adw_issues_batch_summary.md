# ADW Issues Batch Summary Tool Reference

Reads the batch summary table through the issue-batch CLI backend.

Prefer the split wrapper `adw_issues_batch_summary` for all active usage. The
legacy compatibility wrapper `adw_issues_spec` has been retired and archived
under `.trash/`.

## Supported Command

| Command | Required params | Description |
|---|---|---|
| `batch-summary` | `adw_id` | Read the batch summary table |

## Examples

```jsonc
// Preferred split-wrapper usage
{ "adw_id": "abc12345" }

// Historical retired-wrapper equivalent
{ "command": "batch-summary", "adw_id": "abc12345" }
```

## Parameter Reference

| Parameter | Type | Required | Notes |
|---|---|---|---|
| `adw_id` | string | Yes | 8-character hex workflow ID |
| `options` | string | No | No command-scoped option tokens are currently supported |

## Notes

- `batch-summary` keeps its payload-bearing field direct: `adw_id`.
- `options` is accepted for compatibility-surface parity, but no bounded tokens
  are defined for this command.
