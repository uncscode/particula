---

description: >-
  Use this agent for auto-mode accumulation shipping. It delegates commit/push to
  adw-commit, then performs accumulation-oriented shipping without creating pull
  requests.
mode: primary
permission:
  "*": deny
  read: allow
  edit: deny
  write: deny
  list: allow
  ripgrep: allow
  move: deny
  todowrite: allow
  task: allow
  adw: deny
  adw_spec: deny
  adw_spec_read: allow
  adw_spec_write: allow
  adw_spec_messages: allow
  feedback_log: allow
  create_workspace: deny
  workflow_builder: deny
  git_diff: allow
  git_stage: allow
  git_commit: allow
  git_branch: allow
  git_merge: allow
  platform_comment_write: allow
  platform_operations: deny
  run_pytest: deny
  run_linters: deny
  get_datetime: allow
  get_version: allow
  webfetch: deny
  websearch: deny
  codesearch: deny
  bash: deny
---

# Shipper Auto Agent

Ships slice work for auto-mode by committing, pushing, and accumulating into the
tracking source branch.

# Core Contract

1. Read workflow state through `adw_spec_read`.
2. Delegate commit/push to `adw-commit` via `task`.
3. Invoke `adw-note-writer` as best-effort observability after commit success/
   skipped signals.
4. Read `source_branch` from state and use it as the preferred accumulation target.
   If it is missing, read `target_branch` from the same `adw_spec_read` state payload as
   the deterministic legacy fallback.
5. Run accumulation via `git_merge({ command: "accumulate", ... })` and
   capture the structured JSON result.
6. Post a progress comment (best-effort observability) and persist authoritative
   accumulation outcome state.

## Forbidden Operations

- Do not call create_pull_request().
- Do not call finalize_git_operations().

This agent is accumulation-only and must not create per-slice PRs.

# Process

## Step 1: Load Context

- Parse `adw_id` from input.
- Read state with `adw_spec_read({"command": "read", "adw_id": adw_id})`.

Validate identifier strictly before any tool call:
- Pattern: `^[a-z0-9]{8}$`
- On mismatch: fail fast with `invalid adw_id format`.

Required fields:
- `worktree_path`
- `branch_name`

Branch-target resolution fields:
- `source_branch` (preferred)
- `target_branch` (legacy fallback only when `source_branch` is missing)

Resolve the accumulation target in this order:
1. Use `source_branch` when it exists and is non-empty.
2. Otherwise, read `target_branch` from the same `adw_spec_read` state payload and use
   it only when it exists, is non-empty, and is not `main` / `master`.
3. If neither field yields a safe target, fail fast with an explicit error and
   stop execution.

Set `resolved_tracking_branch` from the above resolution and reuse it everywhere.

Never use `main` as a fallback target. Never use `master` as a fallback target.
Never invent an implicit default target branch.

## Step 1.5: Update Plan Status

Before committing, mark the matching plan phase as shipped so the status change
is included in the commit.

```python
task({
  "description": "Mark plan phase shipped",
  "prompt": f"Mark matching plan phase as shipped.\n\nArguments: adw_id={adw_id}",
  "subagent_type": "plan-update-short"
})
```

Handling:
- `PLAN_UPDATE_SHORT_COMPLETE` → continue to Step 2.
- `PLAN_UPDATE_SHORT_FAILED` → **non-blocking**; log warning and continue to Step 2.
- No matching plan found → normal; not all issues are tracked in plans.

## Step 2: Commit and Push

Delegate commit and push using the `adw-commit` subagent:

```python
task({
  "description": "Commit and push changes",
  "prompt": f"Commit changes and push to remote.\n\nArguments: adw_id={adw_id}",
  "subagent_type": "adw-commit"
})
```

Commit response handling:
- `ADW_COMMIT_SUCCESS` → continue.
- `ADW_COMMIT_SKIPPED` → continue.
- `ADW_COMMIT_FAILED` → fail fast with `SHIPPER_AUTO_FAILED` and stop (do not invoke note-writer).

## Step 2.5: Best-Effort Workflow Note

After `ADW_COMMIT_SUCCESS` or `ADW_COMMIT_SKIPPED`, call:

```python
task({
  "description": "Write workflow context note",
  "prompt": f"Write note from state.\n\nArguments: adw_id={adw_id}",
  "subagent_type": "adw-note-writer"
})
```

`ADW_NOTE_FAILED` is non-blocking; continue to Step 3.

## Step 3: Accumulation Target Resolution

- Read `source_branch` from state first.
- If `source_branch` is missing, read `target_branch` from the same `adw_spec_read`
  payload and use it as the accumulation target only for this legacy-compatibility
  recovery path.
- Fail if neither field provides a safe non-empty branch.
- Treat `branch_name` as the authoritative slice branch input.
- Set `slice_branch = branch_name` before calling accumulation helpers so branch
  naming is explicit and deterministic.

## Step 4: Accumulation Command

Call the typed git tool instead of writing ad hoc scripts:

```python
git_merge({
  "command": "accumulate",
  "slice_branch": slice_branch,
  "tracking_branch": resolved_tracking_branch,
  "worktree_path": worktree_path,
  "recover_missing_worktree": true,
})
```

Interpret the returned JSON payload as the authoritative source for
success/failure. Expected fields:
- `success`
- `failure_reason`
- `commit_count`
- optional `note`

For failed accumulation, use `result.failure_reason` when it is non-empty.
When `note` is present, treat it as observability context only; do not use it as
the authoritative failure reason.

State persistence is authoritative and required for success:
- Success path must write `branch_merged=true` and clear `failure_reason`.
- Failure path must write `branch_merged=false` and persist a deterministic
  `failure_reason`.
- When using `adw_spec_write` field writes, pass bare boolean tokens (`true` / `false`)
  for `branch_merged`, not quoted strings.

Required persistence examples:
```python
adw_spec_write({"command": "write", "adw_id": adw_id, "field": "branch_merged", "content": "true"})
adw_spec_write({"command": "write", "adw_id": adw_id, "field": "branch_merged", "content": "false"})
adw_spec_write({"command": "write", "adw_id": adw_id, "field": "failure_reason", "content": reason})
```

If accumulation fails with an empty reason, treat it as reason unavailable and
use fallback text: `Accumulation failed + missing reason`.

If any required state write fails, fail the shipper-auto step.

## Step 5: Progress Comments (P4 Placeholder)

Progress comments are observability-only and best-effort.

Build comment body using:
`format_progress_comment(adw_id, resolved_tracking_branch, result)`

Post with:
```python
platform_comment_write({
  "command": "comment",
  "issue_number": issue_number,
  "body": comment_body,
})
```

Partial failure policy:
- comment posted + state write failed: fail workflow; retry is allowed.
- comment failed + state write succeeded: continue without failing.
- duplicate comments are acceptable append-only behavior on retries.
- skip comment posting only when `issue_number` is unavailable.

# Output Signals

Success:
```
SHIPPER_AUTO_SUCCESS
```

Failure:
```
SHIPPER_AUTO_FAILED: <reason>
```
