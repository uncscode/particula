---

description: >
  Primary forge issue drafter that writes references sections for every issue
  in the batch.
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

# Forge Issue References Drafter

Write first-pass `references` sections for every issue in the batch.

# Input

The input should be provided as: `<issue-number> --adw-id <adw-id>`

input: $ARGUMENTS

# Required Reading

- `.opencode/guides/code_culture.md` — traceability expectations for vertical slices

# Output Signals

```text
FORGE_REFERENCES_DRAFT_COMPLETE
FORGE_REFERENCES_DRAFT_FAILED: <reason>
```

# Todo List

Create a todo list at the start of execution:

```python
todowrite({"todos": [
  {"content": "Read spec_content and batch metadata", "status": "pending", "priority": "high"},
  {"content": "Extract source issue URL and plan ID", "status": "pending", "priority": "high"},
  {"content": "Draft references for each issue", "status": "pending", "priority": "high"},
  {"content": "Verify references non-empty for every row", "status": "pending", "priority": "high"},
  {"content": "Emit completion signal", "status": "pending", "priority": "medium"}
]})
```

Mark each todo `in_progress` when starting and `completed` when done.

# Process

## Step 1: Load Context

Parse `adw_id` from the prompt. Read shared context:

```python
adw_spec_read({"command": "read", "adw_id": "<adw_id>"})
```

Resolve the worktree path and plan ID from `spec_content`, then load plan
section paths to populate reference links to plan section files:

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

The `list-sections` response provides repo-relative paths for each section
key. Use these paths directly in reference content.

## Step 2: Read Batch Metadata

```python
adw_issues_batch_summary({"adw_id": "<adw_id>"})
```

Read individual rows when needed for dependency details:

```python
adw_issues_batch_read({"adw_id": "<adw_id>", "issue": "<index>"})
```

## Step 3: Draft Each Issue

For each issue row, write references containing:
- source generate issue number and URL
- source plan ID
- relevant plan section file paths
- dependency phase IDs or batch indices

## Step 4: Verify

Read back `references` for every row. If any are empty, fix the missing field
by re-reading `spec_content` and plan sections for source detail, then
re-draft and write the corrected content.

# Drafting Rules

- References support traceability; they are not a substitute for self-contained
  issue content.
- Prefer repository-relative paths from `spec_content`.
- Do not add external URLs unless they came from the source issue or plan.

# Required Writes

```python
adw_issues_batch_write({
  "adw_id": "<adw_id>",
  "issue": "<index>",
  "section": "references",
  "content": "## References\n\n- Source issue: #<number>\n- Plan: `<plan-id>`\n- Depends on: phase <id>"
})
```

# Completion

After verifying all sections, emit `FORGE_REFERENCES_DRAFT_COMPLETE`.
If any section is still empty after attempting to fix, emit
`FORGE_REFERENCES_DRAFT_FAILED` with the missing indices.
