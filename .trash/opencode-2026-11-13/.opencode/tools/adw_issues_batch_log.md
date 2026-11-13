# ADW Issues Batch Log Tool Reference

Appends or reads issue-batch review logs through the issue-batch CLI backend.

Prefer the split wrapper `adw_issues_batch_log` for all active usage. The
legacy compatibility wrapper `adw_issues_spec` has been retired and archived
under `.trash/`.

## Supported Command

| Command | Required params | Description |
|---|---|---|
| `batch-log` | `adw_id`, `issue` | Read review logs or append one reviewer result |

## Examples

```jsonc
// Read logs for one row
{ "adw_id": "abc12345", "issue": "1", "options": "read" }

// Append a reviewer result
{ "adw_id": "abc12345", "issue": "1", "reviewer": "testing", "status": "PASS" }

// Append a reviewer result with a note
{ "adw_id": "abc12345", "issue": "1", "reviewer": "scope", "status": "REVISED", "note": "Add touched files." }

// Historical retired-wrapper equivalent
{ "command": "batch-log", "adw_id": "abc12345", "issue": "1", "options": "read" }
```

## Parameter Reference

| Parameter | Type | Required | Notes |
|---|---|---|---|
| `adw_id` | string | Yes | 8-character hex workflow ID |
| `issue` | string | Yes | Positive integer row index |
| `reviewer` | string | Write mode only | Required unless `options` is `read` |
| `status` | string | Write mode only | `PASS` or `REVISED` |
| `note` | string | No | Optional write-mode note |
| `options` | string | No | Bounded command-scoped tokens; `read` is supported |

## Notes

- Keep payload-bearing fields direct: `adw_id`, `issue`, `reviewer`, `status`,
  and `note`.
- Read mode uses `options: "read"`; direct booleans such as `read: true` are
  no longer valid.
