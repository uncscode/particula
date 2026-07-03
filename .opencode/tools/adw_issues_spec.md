# ADW Issues Spec Tool Reference (Retired Compatibility Wrapper)

Historical reference for the retired `adw_issues_spec` compatibility wrapper.

Prefer the split wrappers for new usage:

- `adw_issues_batch_init`
- `adw_issues_batch_read`
- `adw_issues_batch_write`
- `adw_issues_batch_log`
- `adw_issues_batch_summary`

The unified `adw_issues_spec` implementation has been retired from the live
tool tree and archived under `.trash/`.

## Commands

| Command | Required params | Description |
|---|---|---|
| `batch-init` | `total`, `source` | Initialize batch content |
| `batch-read` | `adw_id` | Read metadata, a row, or a section |
| `batch-write` | `adw_id`, `issue`, `content` | Write row or section content |
| `batch-log` | `adw_id`, `issue` | Read logs or append a reviewer result |
| `batch-summary` | `adw_id` | Read the summary table |

## Examples

```jsonc
// Initialize a batch
{ "command": "batch-init", "total": "5", "source": "docs/feature.md" }

// Read one row section as raw text
{ "command": "batch-read", "adw_id": "abc12345", "issue": "1", "section": "scope", "options": "raw" }

// Write section content
{ "command": "batch-write", "adw_id": "abc12345", "issue": "1", "section": "testing_strategy", "content": "## Tests\n- add coverage" }

// Read review logs
{ "command": "batch-log", "adw_id": "abc12345", "issue": "1", "options": "read" }

// Append review feedback
{ "command": "batch-log", "adw_id": "abc12345", "issue": "1", "reviewer": "testing", "status": "PASS", "note": "Looks good." }

// Read summary table
{ "command": "batch-summary", "adw_id": "abc12345" }
```

## Bounded `options` Tokens

| Command | Supported tokens |
|---|---|
| `batch-read` | `raw` |
| `batch-log` | `read` |
| `batch-init`, `batch-write`, `batch-summary` | none |

## Direct Payload-Bearing Fields

Keep these fields direct instead of routing them through `options`:

- `adw_id`
- `issue`
- `section`
- `content`
- `reviewer`
- `status`
- `note`
- `total`
- `source`

## Notes

- Bounded tokens are bare words such as `options: "raw"` or
  `options: "read"`.
- Removed direct booleans such as `raw: true` and `read: true` are not part of
  the current wrapper contract.
