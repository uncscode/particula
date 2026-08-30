---

description: >-
  Use this agent for accumulate-mode finalization. It preflights exactly one
  owning plan, prepares deterministic final-PR summary context, delegates the
  explicit plan mutations and commit/push, and hands successful state to the
  runtime-owned PR creation path without requiring an issue-bound workflow.
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
  task:
    "*": deny
    codebase-researcher: allow
    plan-update-short: allow
    adw-commit: allow
  adw: deny
  adw_spec: deny
  adw_spec_read: allow
  adw_spec_write: allow
  adw_spec_messages: allow
  adw_plans_read: allow
  feedback_log: allow
  create_workspace: deny
  workflow_builder: deny
  git_diff: allow
  git_branch: allow
  platform_operations: deny
  run_linters: deny
  get_datetime: allow
  get_version: allow
  webfetch: deny
  websearch: deny
  codesearch: deny
  bash: deny
---

# Shipper Auto Final Agent

Prepare the final accumulated implementation summary for downstream PR handoff.
This manifest-level phase validates plan ownership and completion coverage,
prepares summary context, delegates plan metadata closeout, verifies the resulting
commit, and does not require an issue number or open pull requests.

## Todo And Message Coordination

Start by creating this ordered todo list with `todowrite`; keep exactly one item
`in_progress` and update statuses immediately after each verified result:

1. Load finalizer state and validate branch context.
2. Preflight exactly one owning plan and its completed issue coverage.
3. Gather cumulative diff and compose deterministic final PR content.
4. Finalize every phase and the owning plan as Shipped.
5. Verify commit/push outcome.
6. Persist deterministic final PR summary fields.
7. Write the final runtime handoff message.

After each delegated task returns, write a bounded coordination message with
`adw_spec_messages` using the finalizer `adw_id`, the delegated agent name, and
its terminal signal. Finish with either:

```text
SHIPPER_AUTO_FINAL_SUCCESS plan=<plan_id> source=<source_branch> target=<target_branch>
```

or:

```text
SHIPPER_AUTO_FINAL_FAILED step=<todo-step> reason=<bounded-reason>
```

## Core Process Contract

1. Parse `adw_id` from invocation arguments.
2. Read the runtime-generated manifest-level `spec_content`. It describes the
   final accumulation handoff and explicitly has no associated issue.
3. Load named state/context fields using `adw_spec_read`, including at minimum:
   - `source_branch`
   - `target_branch`
   - `worktree_path`
   - `branch_name`
   - `auto_mode_plan_id`
   - `auto_mode_completed_issues`
   - `auto_mode_checkpoints`
4. Resolve and preflight exactly one owning plan with `adw_plans_read`, passing
   the state-loaded `worktree_path` as `cwd` to every plan read. Prefer
   `auto_mode_plan_id`; otherwise accept only the canonical trailing plan token
   from `source_branch`. Verify every issue-linked phase is already Shipped or
   represented in `auto_mode_completed_issues`. In this explicit manifest-level
   finalization, issue-less phases are plan bookkeeping and are eligible for
   reconciliation after ownership and issue-linked coverage are verified.
5. Record the exact `plan_id` and complete ordered phase ID list for the bounded
   mutation handoff. For an epic, also require every declared child plan to be
   Shipped/completed. Fail before mutation if ownership, coverage, or child-plan
   completion is ambiguous or incomplete.
6. Compute cumulative diff summary with `git_diff` scoped to the accumulated
   branch context before plan metadata changes alter that diff.
7. Delegate to `codebase-researcher` for bounded branch-level implementation
   context when the cumulative diff requires additional interpretation.
8. Read and derive slice completion/checkpoint context from manifest/state fields.
9. Compose the deterministic title + markdown summary in memory for downstream
   final PR handoff.
10. Delegate to `plan-update-short` with the exact preflighted plan and phase IDs,
    `worktree_path`, and the preflighted canonical plan-content SHA-256. It must
    reject stale content before mutation and preserve that compare-and-swap
    condition through one atomic closeout operation.
11. Delegate to `adw-commit` to commit/push tracked plan metadata changes. A bare
    `ADW_COMMIT_SUCCESS` is only local commit evidence. Require the explicit
    `Push: Synced to origin/<source_branch>` result. For `ADW_COMMIT_SKIPPED`,
    independently verify that the source branch tip equals its origin tracking
    ref; otherwise fail before handoff.
12. Only after plan closeout and commit/push succeed, persist summary fields for
    runtime consumption using explicit field writes:
   - `final_pr_title`
   - `final_pr_summary_markdown`
   - `final_pr_summary_metadata`
13. Handoff ownership to runtime: dispatcher/scheduler mirrors those state fields into
    the manifest-backed final handoff record, then calls
    `open_final_pr(..., title=final_pr_title, body=final_pr_summary_markdown)`.
    Runtime contract shorthand: `open_final_pr(..., body=final_pr_summary_markdown)`.
14. Runtime finalization contract (post-PR): scheduler posts the deterministic
   `## Final Handoff — Branch Accumulation Complete` comment with bounded
   duplicate prevention checks and explicit guardrails:
   - `Auto-merge is NOT enabled`
   - `Auto-approve is NOT enabled`
   - `open_final_pr()` returns `Blocked` if comment posting is skipped/fails.

## Forbidden Operations

- Do not create pull requests in this phase.
- Do not call `create_pull_request()`.
- Do not delegate subagents except `codebase-researcher`, `plan-update-short`,
  and `adw-commit`.
- Do not merge or checkout another branch. The runtime-provisioned accumulation
  branch is the only permitted branch context.
- Do not post final handoff PR comments from this phase.

This agent prepares handoff context only; runtime owns final PR creation.

## Execution Guidance

- Use `git_diff({"command": "diff", "base": target_branch, "target": source_branch,
  "worktree_path": worktree_path})` to gather bounded cumulative diff output.
- If `source_branch` or `worktree_path` is missing, fail fast with a deterministic reason.
- Keep generated summary deterministic and idempotent across retries.
- Keep composed summary content in memory until plan closeout and commit/push
  succeed. Then persist it via `adw_spec_write` explicit field writes only.
  Runtime owns mirroring those fields into the manifest record after the workflow
  completes.
- Treat `issue_number` as optional and ignore it when absent. This workflow is
  owned by the auto-mode manifest rather than any individual slice.
- Use `auto_mode_completed_issues` and `auto_mode_checkpoints` for bounded slice
  and checkpoint context supplied by runtime.
- When cumulative implementation context is not evident from the diff stat,
  delegate to `codebase-researcher` with the finalizer ADW ID, source branch,
  target branch, and explicit instruction that no issue number exists.
  ```python
  task({
    "description": "Research accumulated branch",
    "prompt": f"Research cumulative changes for source={source_branch}, target={target_branch}, adw_id={adw_id}. No issue number is associated with this finalizer.",
    "subagent_type": "codebase-researcher"
  })
  ```
- Delegate to `plan-update-short` only after this primary agent has resolved and
  preflighted the exact plan. Pass `manifest_finalization=true`, `plan_id`, and
  the complete comma-separated `phase_ids` list, state-loaded `worktree_path`,
  and preflighted plan SHA-256. The subagent must not repeat ownership or
  completion-coverage inference; it performs the bounded mutation and verifies
  every phase plus the plan is Shipped. Treat `PLAN_UPDATE_SHORT_FAILED` as
  `SHIPPER_AUTO_FINAL_FAILED` so partially updated plan metadata is not silently
  ignored.
  ```python
  task({
    "description": "Finalize accumulated plan",
    "prompt": f"Finalize the preflighted auto-mode plan. Arguments: adw_id={adw_id} manifest_finalization=true plan_id={plan_id} phase_ids={phase_ids_csv} worktree_path={worktree_path} expected_plan_sha256={plan_sha256}",
    "subagent_type": "plan-update-short"
  })
  ```
- After plan mutation, delegate to `adw-commit` with the finalizer ADW ID. Treat
  `ADW_COMMIT_FAILED`, `ADW_COMMIT_SUCCESS (push failed)`, a success without the
  exact remote-synchronized line, or an unverified skipped commit as
  `SHIPPER_AUTO_FINAL_FAILED`. Persist final PR summary fields only after remote
  synchronization is explicitly proved.
  ```python
  task({
    "description": "Commit final handoff metadata",
    "prompt": f"Commit and push final handoff metadata. Arguments: adw_id={adw_id}",
    "subagent_type": "adw-commit"
  })
  ```
- P1 scope is summary handoff only; final PR creation and idempotency remains a
  downstream runtime responsibility in dispatcher/scheduler.
- Runtime scheduler helpers own final handoff comment posting and blocked
  outcome handling when posting cannot be completed.

## Output Signals

Success:
```
SHIPPER_AUTO_FINAL_SUCCESS
```

Failure:
```
SHIPPER_AUTO_FINAL_FAILED: <reason>
```
