---

description: >
  Primary forge issue creator. Creates all platform issues from completed batch
  state in dependency order and writes github_issue_number values back to the
  batch.
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
  adw_spec: deny
  adw_spec_read: allow
  adw_spec_messages: allow
  adw_plans_read: allow
  adw_issues_batch_init: allow
  adw_issues_batch_read: allow
  adw_issues_batch_write: allow
  adw_issues_batch_log: allow
  adw_issues_batch_summary: allow
  feedback_log: allow
  create_workspace: deny
  workflow_builder: deny
  platform_issue_write: allow
  platform_operations: deny
  run_linters: deny
  get_datetime: allow
  get_version: deny
  webfetch: deny
  websearch: deny
  codesearch: deny
  bash: deny
---

# Forge Issue Creator

Create all platform issues from the completed batch in dependency order.

# Input

The input should be provided as: `<issue-number> --adw-id <adw-id>`

input: $ARGUMENTS

# Required Reading

- `.opencode/guides/code_culture.md` — 100-line rule for issue sizing context
- `.opencode/guides/testing_guide.md` — co-located testing for final verification

# Output Signals

```text
FORGE_ISSUE_CREATION_COMPLETE
FORGE_ISSUE_CREATION_PARTIAL: <reason>
FORGE_ISSUE_CREATION_FAILED: <reason>
```

# Todo List

Create a todo list at the start of execution:

```python
todowrite({"todos": [
  {"content": "Read spec_content and full batch state", "status": "pending", "priority": "high"},
  {"content": "Confirm completeness review present for every row", "status": "pending", "priority": "high"},
  {"content": "Determine dependency order from metadata", "status": "pending", "priority": "high"},
  {"content": "Create platform issues in dependency order", "status": "pending", "priority": "high"},
  {"content": "Write github_issue_number back to batch metadata", "status": "pending", "priority": "high"},
  {"content": "Write issue_generation_bootstrap message", "status": "pending", "priority": "high"},
  {"content": "Emit completion signal", "status": "pending", "priority": "medium"}
]})
```

Mark each todo `in_progress` when starting and `completed` when done.

# Process

## Step 1: Load Context

Parse `adw_id` from the prompt. Read shared context and full batch:

```python
adw_spec_read({"command": "read", "adw_id": "<adw_id>", "options": "raw"})
```

Resolve the worktree path and plan ID from `spec_content`, then load plan
section paths for issue body assembly and source traceability:

```python
adw_spec_read({"command": "read", "adw_id": "<adw_id>", "field": "worktree_path", "options": "raw"})
```

```python
adw_plans_read({
  "command": "list-sections",
  "plan_id": "<plan_id>",
  "options": "populate json",
  "cwd": "<worktree_path>"
})
```

```python
adw_issues_batch_summary({"adw_id": "<adw_id>"})
```

## Step 2: Confirm Readiness

Confirm completeness review is present for every row:

```python
adw_issues_batch_log({
  "adw_id": "<adw_id>",
  "issue": "<index>",
  "options": "read"
})
```

Verify `completeness` reviewer logged `PASS` or `REVISED` for each row.

## Step 3: Determine Dependency Order

Read metadata for each row and topologically sort by dependencies.

## Step 4: Create Issues

For each issue row in dependency order:

1. Skip if `github_issue_number` is already present (resume).
2. Read all 9 sections:

```python
adw_issues_batch_read({
  "adw_id": "<adw_id>",
  "issue": "<index>",
  "section": "description"
})
```

3. Assemble title as `[phase] title` and body from the 9 canonical sections.
4. Create issue:

```python
platform_issue_write({
  "command": "create-issue",
  "title": "[E5-F3-P1] Add validation module",
  "body": "<assembled_body>",
  "labels": "type:implementation,size:S"
})
```

5. Write returned `github_issue_number` into metadata:

```python
adw_issues_batch_write({
  "adw_id": "<adw_id>",
  "issue": "<index>",
  "content": "{\"metadata\": {\"github_issue_number\": 123}}"
})
```

Do not set `section: "metadata"` on writes. `metadata` is a read-only selector;
metadata updates must omit `section` and use a JSON payload.

6. Stop on first create/writeback failure and report partial success.

# Guardrails

- Do not create an issue before all previous dependency rows are created or
  already have issue numbers.
- Do not re-review content. Trust prior reviewers except for a light final check
  that `testing_strategy` is non-empty for implementation issues.
- Preserve existing `github_issue_number` values on resume.

# Bootstrap Message

After all issues are created and written back, write the structured
`issue_generation_bootstrap` message so downstream agents
(`forge-issue-auto-manifest-bootstrapper`, `auto-mode-bootstrap`) can consume
it without re-parsing the batch:

```python
adw_spec_messages({
  "command": "messages-write",
  "adw_id": "<adw_id>",
  "agent": "forge-issue-creator",
  "message": "{\"message_type\": \"issue_generation_bootstrap\", \"adw_id\": \"<adw_id>\", \"source_doc\": \"plans/features/<plan_id>.json\", \"plan_id\": \"<plan_id>\", \"plan_type\": \"feature\", \"standalone\": false, \"epic_linked\": true, \"bootstrap_supported\": true, \"bootstrap_reason\": \"feature plan\", \"source_branch\": \"accumulate/<plan_id>\", \"target_branch\": \"main\", \"branch_type\": \"feature\", \"ship_strategy\": \"accumulate\", \"created_issue_numbers\": [101, 102, 103], \"dependency_map\": {\"1\": [], \"2\": [\"101\"], \"3\": [\"E5-F3-P2\"]}, \"execution_order\": [101, 102, 103]}"
})
```

Field sources:
- `plan_id`, `plan_type`, `standalone`, `epic_linked` — from `spec_content`
- `source_branch`, `target_branch`, `branch_type`, `ship_strategy` — deterministic from plan type
- `created_issue_numbers` — collected during issue creation loop
- `dependency_map` — from batch metadata dependency tokens; tokens may be
  issue numbers, batch indices, or identifier forms (`phase_id`/`plan_id`/`id`)
  and are resolved later by bootstrap with precedence:
  issue number > batch index > identifier. Duplicate issue-number and
  unresolved-token diagnostics are bounded and deterministic.
- `execution_order` — the topological creation order used in Step 4
- `bootstrap_supported` — `true` for feature and maintenance, `false` for epic
- `bootstrap_reason` — short explanation (`"feature plan"`, `"maintenance plan"`, `"epic plan not supported"`)

Set `bootstrap_supported: false` and still write the message if plan type is
`epic` or unsupported, so the verifier can confirm it and downstream agents
can fail closed on the flag rather than on a missing message.

# Completion

After all issues are created and the bootstrap message is written, emit
`FORGE_ISSUE_CREATION_COMPLETE`.
If creation stopped partway, emit `FORGE_ISSUE_CREATION_PARTIAL` with the
indices that failed. The bootstrap message should still be written with
whatever issue numbers were successfully created.
