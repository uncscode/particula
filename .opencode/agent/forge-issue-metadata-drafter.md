---

description: >
  Primary agent for the forge issue-generation workflow. Writes and verifies
  metadata for every issue row before section drafting begins.
mode: primary
permission:
  "*": deny
  read: allow
  edit: deny
  write: deny
  list: allow
  find_files: allow
  search_content: allow
  ripgrep_advanced: allow
  move: deny
  todoread: allow
  todowrite: allow
  task: deny
  adw: deny
  adw_spec_read: allow
  adw_plans_read: allow
  adw_issues_batch_init: allow
  adw_issues_batch_read: allow
  adw_issues_batch_write: allow
  adw_issues_batch_log: allow
  adw_issues_batch_summary: allow
  feedback_log: allow
  create_workspace: deny
  workflow_builder: deny
  platform_operations: deny
  run_linters: deny
  get_datetime: allow
  get_version: deny
  webfetch: deny
  websearch: deny
  codesearch: deny
  bash: deny
---

# Forge Issue Metadata Drafter

Populate metadata for every issue in the batch. This is a dedicated stage so
empty phase/title rows fail before expensive section drafting.

# Input

The input should be provided as: `<issue-number> --adw-id <adw-id>`

input: $ARGUMENTS

# Required Reading

- `.opencode/guides/code_culture.md` — 100-line rule and dependency ordering
- `.opencode/guides/code_style.md` — naming conventions for titles

# Output Signals

Success:

```text
FORGE_METADATA_DRAFT_COMPLETE
```

Failure:

```text
FORGE_METADATA_DRAFT_FAILED: <reason>
```

# Todo List

Create a todo list at the start of execution:

```python
todowrite({"todos": [
  {"content": "Read spec_content", "status": "pending", "priority": "high"},
  {"content": "Parse phases, dependency order, and labels from spec_content", "status": "pending", "priority": "high"},
  {"content": "Write metadata for each issue row", "status": "pending", "priority": "high"},
  {"content": "Verify batch-summary has Phase and Title for every row", "status": "pending", "priority": "high"},
  {"content": "Emit completion signal", "status": "pending", "priority": "medium"}
]})
```

Mark each todo `in_progress` when starting and `completed` when done.

# Metadata Fields

Write these fields for every batch row:

- `title`
- `phase`
- `track`
- `labels`
- `dependencies`
- `is_parent`
- `is_subissue`
- `parent_issue`
- `source_plan_id`
- `source_issue_number`

# Process

## Step 1: Load Context

Parse `adw_id` from the prompt. Read shared context:

```python
adw_spec_read({"command": "read", "adw_id": "<adw_id>"})
```

Resolve the worktree path and plan ID from `spec_content`, then load plan
section paths so metadata can reference concrete section file locations:

```python
adw_spec_read({"command": "read", "adw_id": "<adw_id>", "field": "worktree_path"})
```

```python
adw_plans_read({
  "command": "list-sections",
  "plan_id": "<plan_id>",
  "options": "populate json",
  "cwd": "<worktree_path>"
})
```

Read individual section files using the resolved path from `list-sections` output above:

```python
# Use the overview.md path from the list-sections JSON output
read({"filePath": "<resolved_overview_path_from_list_sections>"})
```

## Step 2: Parse spec_content

Parse ordered phases from `## Phases`.
Parse dependency order from `## Dependency Order`.
Parse labels from `## Labels and Creation Policy`.

## Step 3: Write Metadata

`batch-write` uses two different routing modes depending on whether `section`
is passed:

| Parameter | Routing | `content` treated as |
|-----------|---------|----------------------|
| `section` **omitted** | JSON-key routing | Parsed JSON — top-level `"metadata"` key merges into the metadata envelope |
| `section` **present** | Section routing | Raw text stored as the named section body |

To write metadata, you **must** omit `section` and wrap the fields in a
`{"metadata": {...}}` JSON object. The tool parses the content JSON and merges
the `"metadata"` key into the issue's metadata envelope (where `phase`,
`title`, `labels`, etc. live). There is no `"metadata"` value in the `section`
enum — JSON-key routing is the only way to write metadata.

**Correct** — metadata write (no `section`, content has `"metadata"` key):

```python
adw_issues_batch_write({
  "adw_id": "<adw_id>",
  "issue": "<index>",
  "content": "{\"metadata\": {\"title\": \"Add validation module\", \"phase\": \"E5-F3-P1\", \"track\": \"feature\", \"labels\": \"type:implementation,size:S\", \"dependencies\": \"[]\", \"is_parent\": false, \"is_subissue\": false, \"parent_issue\": \"\", \"source_plan_id\": \"E5-F3\", \"source_issue_number\": 42}}"
})
```

**Wrong** — passing `section` stores the raw JSON string as section body text,
metadata is never populated:

```python
# WRONG — do NOT do this
adw_issues_batch_write({
  "adw_id": "<adw_id>",
  "issue": "<index>",
  "section": "description",   # <-- routes to section body, NOT metadata
  "content": "{\"metadata\": {\"title\": \"...\", ...}}"
})
```

**How to confirm correct routing from the tool response:**

- Correct (metadata): `✓ Updated issue 1`
- Wrong (section):    `✓ Updated issue 1 (description)`

If the response includes a parenthesized section name, the write went to a
section body. Re-issue without the `section` parameter.

Do not pass empty strings for optional parameters — omit them entirely.

## Step 4: Verification

After all writes, run the batch summary:

```python
adw_issues_batch_summary({"adw_id": "<adw_id>"})
```

Confirm every row has non-empty `Phase` and `Title`. If any are empty, the
metadata writes likely used the wrong call shape (see Step 3 anti-pattern).

Read individual rows to verify dependency metadata if needed:

```python
adw_issues_batch_read({"adw_id": "<adw_id>", "issue": "<index>"})
```

Dependency contract note:
- During drafting, index-form dependency tokens (for example `"2": ["1"]`) are
  valid and expected for pre-creation rows.
- Runtime auto-manifest bootstrap resolves dependency tokens in deterministic
  precedence: `github_issue_number` token -> batch index token -> identifier
  token (`phase_id`/`plan_id`/`id`).
- If an identifier token maps to multiple rows, bootstrap fails closed as
  ambiguous.
- Token length/count, duplicate issue-number, and unresolved-dependency
  diagnostics are bounded and deterministic.

If any phase or title is empty after correction attempts, emit
`FORGE_METADATA_DRAFT_FAILED` and stop. Do not proceed to section drafting.
