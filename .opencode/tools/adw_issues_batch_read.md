# ADW Issues Batch Read Tool Reference

Reads issue-batch content through the issue-batch CLI backend.

Prefer the split wrapper `adw_issues_batch_read` for all active usage. The
legacy compatibility wrapper `adw_issues_spec` has been retired and archived
under `.trash/`.

## Supported Command

| Command | Required params | Description |
|---|---|---|
| `batch-read` | `adw_id` | Read batch metadata, an issue row, or a specific section |

## Examples

```jsonc
// Read metadata / full batch state
{ "adw_id": "abc12345" }

// Read one row
{ "adw_id": "abc12345", "issue": "1" }

// Read metadata for one row
{ "adw_id": "abc12345", "issue": "1", "section": "metadata" }

// Read a section for one row
{ "adw_id": "abc12345", "issue": "1", "section": "scope" }

// Raw output uses bounded options tokens
{ "adw_id": "abc12345", "issue": "1", "section": "scope", "options": "raw" }

// Read full batch state as raw JSON
{ "adw_id": "abc12345", "options": "raw" }

// Historical retired-wrapper equivalent
{ "command": "batch-read", "adw_id": "abc12345", "issue": "1", "options": "raw" }
```

## Parameter Reference

| Parameter | Type | Required | Notes |
|---|---|---|---|
| `adw_id` | string | Yes | 8-character hex workflow ID |
| `issue` | string | No | Positive integer row index |
| `section` | string | No | Non-empty section token |
| `options` | string | No | Bounded command-scoped tokens; `raw` is supported |

## Notes

- Keep payload-bearing fields direct: `adw_id`, `issue`, and `section`.
- `section: "metadata"` is a supported read-only selector for issue metadata.
- `options` accepts only bare tokens such as `raw`; `raw=true` is not valid.
