# Auto Mode Manifest Tool Reference

Structured wrapper for auto-mode manifest operations.

## Commands

| Command | Required params | Description |
|---|---|---|
| `init-from-batch` | `adw_id` | Build a manifest from issue-batch state |
| `init` | `issues` | Build a manifest from explicit issue numbers |
| `status` | — | Show manifest state |
| `validate` | — | Validate manifest state |
| `delete` | `branch` | Remove one manifest registry entry by exact source branch |
| `prune` | `completed=true` | Remove all registry entries whose top-level status is `completed` |
| `reset` | `issue` | Reset one issue's manifest state |
| `complete` | `issue`, `adw_id` | Mark one issue completed for accumulate-mode handoff |

## Preferred Usage

This is the active compatibility wrapper for auto-mode manifest operations until
the narrower split replacement path ships. Optional toggles now use bounded
command-scoped `options` tokens, while payload-bearing fields stay direct.

## Examples

```jsonc
// Initialize from batch state
{ "command": "init-from-batch", "adw_id": "abc12345", "ship_strategy": "accumulate", "segment_size": 3, "options": "force" }

// Initialize from an explicit issue list
{ "command": "init", "issues": "42,43", "depends": "43:42", "ship_strategy": "pr", "options": "force" }

// Read JSON status
{ "command": "status", "branch": "feature/F42", "options": "json" }

// Validate a branch-scoped manifest
{ "command": "validate", "branch": "feature/F42" }

// Preview deleting exactly one manifest registry entry
{ "command": "delete", "branch": "feature/F42", "options": "dry-run" }

// Force-prune all manifests whose top-level status is completed
{ "command": "prune", "completed": true, "options": "force" }

// Reset an issue and resume downstream work
{ "command": "reset", "issue": "42", "branch": "feature/F42", "options": "resume force" }

// Complete an issue without PR/commit actions
{ "command": "complete", "issue": "42", "adw_id": "abc12345", "branch": "feature/F42", "completed_at": "2026-06-27T23:59:59Z", "detail": "Issue completed (branch accumulation).", "options": "branch-merged dry-run" }
```

## Bounded `options` Tokens

| Command | Supported tokens |
|---|---|
| `init-from-batch` | `force` |
| `init` | `force` |
| `status` | `json` |
| `validate` | none |
| `delete` | `dry-run`, `force` |
| `prune` | `dry-run`, `force` |
| `reset` | `resume`, `force` |
| `complete` | `force`, `dry-run`, `branch-merged`, `no-branch-merged` |

## Direct Payload-Bearing Fields

Keep these fields direct:

- `adw_id`
- `issues`
- `depends`
- `issue`
- `source_branch`
- `target_branch`
- `branch_type`
- `segment_size`
- `ship_strategy`
- `branch`
- `completed`
- `completed_at`
- `detail`

## Notes

- Use bare tokens such as `options: "force"`, `options: "json"`, or
  `options: "resume force"`.
- Wrapper-routed `delete` and `prune` calls must stay non-interactive: include
  `options: "dry-run"`, `options: "force"`, or both. Calls that omit both are
  rejected before spawn so agents never hang on a confirmation prompt.
- Wrapper-routed `branch` inputs for `status`, `validate`, `delete`, `reset`,
  and `complete` must be non-blank after trimming; whitespace-only values are
  rejected before spawn.
- `delete` and `prune` remove manifest registry entries only. They do **not**
  delete Git branches, remotes, or worktrees; mutate platform issue state; or
  touch per-workflow ADW state.
- `prune` selects only manifests whose top-level `status == "completed"`.
  This is intentionally different from cron-owned cleanup lifecycle pruning,
  which also considers retained `cleanup_status` state.
- Cleanup lifecycle note for accumulate mode:
  - `status` and `cleanup_status` are different surfaces. `status=completed`
    means issue accumulation is done; retained cleanup visibility lives in
    `cleanup_status` (`not_started`, `awaiting_final_pr`, `awaiting_merge`, `ready`,
    `running`, `failed_retryable`, `blocked_operator_action`, `complete`,
    `safe_skipped`).
  - New or just-completed accumulate manifests default to
    `cleanup_status=not_started` until persisted final-PR or cleanup evidence
    advances them.
  - `completed` records issue completion/branch accumulation state; later hourly
    cleanup may reconcile a merged final PR into retained `ready`, then proceed
    through branch/worktree/local cleanup before pruning.
  - `reset` is the operator recovery path when retained cleanup state is
    blocked and manual intervention is required before another pass.
  - Only terminal cleanup states `complete` or `safe_skipped` are pruneable;
    retained states such as `awaiting_final_pr`, `awaiting_merge`, `ready`,
    `running`, `failed_retryable`, and `blocked_operator_action` remain visible
    until cleanup progresses or an operator intervenes.
  - See `docs/Examples/operations/auto-mode-runbook.md` for full cadence,
    prerequisites, visible failure states, and recovery guidance.
- `complete` defaults `branch_merged` to true and completes running or fixing
  issues by default; use `force` for other states.
- Removed direct booleans such as `force: true`, `json: true`, and
  `resume: true` are not part of the current contract.
- `segment_size` accepts a number or numeric string; blank or whitespace-only
  values are treated as omitted.
- `metadata.dependencies` accepted by `init-from-batch` may contain either a
  native list or a JSON-encoded list string; malformed values fail closed.
