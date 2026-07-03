# ADW Issues Batch Write Tool Reference

Writes issue-batch content through the issue-batch CLI backend.

Prefer the split wrapper `adw_issues_batch_write` for all active usage. The
legacy compatibility wrapper `adw_issues_spec` has been retired and archived
under `.trash/`.

## Supported Command

| Command | Required params | Description |
|---|---|---|
| `batch-write` | `adw_id`, `issue`, `content` | Write a row payload or section content |

## Examples

```jsonc
// Write a whole row payload
{ "adw_id": "abc12345", "issue": "1", "content": "{\"metadata\": {\"title\": \"Draft\"}, \"sections\": {}}" }

// Write one section
{ "adw_id": "abc12345", "issue": "1", "section": "testing_strategy", "content": "## Tests\n- add coverage" }

// Write metadata only by omitting section and sending a JSON merge payload
{ "adw_id": "abc12345", "issue": "1", "content": "{\"metadata\": {\"github_issue_number\": 123}}" }

// Historical retired-wrapper equivalent
{ "command": "batch-write", "adw_id": "abc12345", "issue": "1", "section": "scope", "content": "Touched files..." }
```

## Parameter Reference

| Parameter | Type | Required | Notes |
|---|---|---|---|
| `adw_id` | string | Yes | 8-character hex workflow ID |
| `issue` | string | Yes | Positive integer row index |
| `content` | string | Yes | Payload to write |
| `section` | string | No | Optional section token |
| `options` | string | No | No bounded option tokens are currently supported |

## Notes

- Keep payload-bearing fields direct: `adw_id`, `issue`, `content`, `section`.
- `section: "metadata"` is not valid for writes. To update metadata, omit
  `section` and send a JSON object with a `metadata` key.
- `batch-write` accepts `options` only for surface parity; no command-scoped
  tokens are currently defined.
