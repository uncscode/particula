# Platform Operations Tool Reference

`platform_operations` is a compatibility/delegation path.
Use split wrappers for migrated commands.

## Split Wrapper Mapping (Primary Path)

| Command | Primary wrapper |
|---|---|
| `create-pr` | `platform_pr_write` |
| `pr-comments` | `platform_pr_read` |
| `fetch-issue` | `platform_issue_read` |
| `create-issue` | `platform_issue_write` |
| `update-issue` | `platform_issue_write` |
| `add-labels` | `platform_label_write` |
| `remove-labels` | `platform_label_write` |
| `rate-limit` | `platform_rate_limit_read` |
| `comment` | `platform_comment_write` |
| `pr-review` | `platform_pr_review_write` |

## Compatibility Command Surface

| Command | Required params | Notes |
|---|---|---|
| `create-pr` | `title`, `head` | Delegates to split wrapper path |
| `fetch-issue` | `issue_number` | Delegates to split wrapper path |
| `create-issue` | `title` | Delegates to split wrapper path |
| `update-issue` | `issue_number` + 1 mutable field | Delegates to split wrapper path |
| `add-labels` | `issue_number`, `labels` | Delegates to split wrapper path |
| `remove-labels` | `issue_number`, `labels` | Delegates to split wrapper path |
| `pr-comments` | `issue_number` | Delegates to split wrapper path |
| `rate-limit` | — | Delegates to split wrapper path |
| `comment` | `issue_number`, `body` | Delegates to split wrapper path |
| `pr-review` | `issue_number`, `body` | Delegates to split wrapper path |

## Preferred Examples

Use split wrappers directly for migrated commands.

```jsonc
// create-pr (preferred)
platform_pr_write({"command": "create-pr", "title": "feat: #123 - Add auth", "head": "feature-123", "adw_id": "abc12345"})

// fetch issue (preferred)
platform_issue_read({"command": "fetch-issue", "issue_number": "123", "output_format": "json"})

// create/update issue (preferred)
platform_issue_write({"command": "create-issue", "title": "Bug: login fails", "body": "Details..."})
platform_issue_write({"command": "update-issue", "issue_number": "123", "state": "closed"})

// labels (preferred)
platform_label_write({"command": "add-labels", "issue_number": "123", "labels": "enhancement,docs"})

// rate limit (preferred)
platform_rate_limit_read({"command": "rate-limit", "output_format": "json"})

// comment (preferred)
platform_comment_write({"command": "comment", "issue_number": "123", "body": "LGTM"})

// pr-review (preferred)
platform_pr_review_write({"command": "pr-review", "issue_number": "42", "body": "Looks good overall"})
```

## Deterministic Failure and Recovery Examples

Pre-spawn validation (missing required identifier):

```text
ERROR: pr-comments command requires issue_number
```

Delegated/subprocess failure envelope:

```text
ERROR: Failed to execute 'adw platform <command>'
```

Diagnostic precedence is deterministic: `stderr` -> `stdout` -> `message/fallback`.

Recovery / routing hint:

- Use split wrappers directly for migrated operations (`platform_comment_write`, `platform_pr_review_write`, etc.).
- Keep `platform_operations` for compatibility/delegation flows only.
