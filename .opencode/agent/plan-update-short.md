---
description: >
  Subagent that marks plan phases as shipped and updates plan lifecycle/status
  after workflow completion. It supports both issue-linked phase updates and
  issue-less auto-mode manifest finalization.

  This subagent:
  - Loads workflow context from adw_spec_read
  - Resolves a plan by issue number or manifest finalization context
  - Uses adw_plans_mutate update-phase to mark one or all phases Shipped
  - Uses adw_plans_mutate update to promote plan status if all phases are done
  - Runs during the shipping step of most workflows

  Invoked by: shipper, shipper-auto, or documentation primary agent
mode: subagent
permission:
  "*": deny
  read: allow
  grep: allow
  find_files: allow
  search_content: allow
  ripgrep_advanced: allow
  todowrite: allow
  adw_spec_read: allow
  adw_spec_messages: allow
  adw_plans_read: allow
  adw_plans_mutate: allow
  feedback_log: allow
  get_datetime: allow
  get_version: allow
---

# Plan Update Short Subagent

Mark plan phases as shipped and update plan lifecycle after workflow completion.

# Core Mission

After a workflow ships, find the matching plan phase by issue number and mark it
as Shipped. For issue-less auto-mode finalization, resolve exactly one owning
plan and mark every phase Shipped. Promote the plan only after verifying every
phase is Shipped.

# Input Format

```
Arguments: adw_id=<workflow-id>
```

Manifest finalization adds `manifest_finalization=true`.

**Invocation:**
```python
task({
  "description": "Mark plan phase shipped",
  "prompt": f"Mark matching plan phase as shipped.\n\nArguments: adw_id={adw_id}",
  "subagent_type": "plan-update-short"
})
```

# Process

## Step 1: Load Context

First list state fields, then read the named fields required by the selected
mode. Do not assume an issue-derived spec is present for normal compatibility
paths; manifest finalization receives a runtime-generated general spec.

```python
adw_spec_read({"command": "list", "adw_id": "{adw_id}"})
adw_spec_read({"command": "read", "adw_id": "{adw_id}", "field": "source_branch"})
adw_spec_read({"command": "read", "adw_id": "{adw_id}", "field": "auto_mode_plan_id"})
adw_spec_read({
  "command": "read",
  "adw_id": "{adw_id}",
  "field": "auto_mode_completed_issues"
})
```

Extract:
- `issue_number` - The issue that just shipped, when present
- `worktree_path` - ADW worktree root for all plan tool calls
- `source_branch` - Accumulation branch for manifest finalization
- `auto_mode_plan_id` - Runtime-derived plan ID when the branch encodes one
- `auto_mode_completed_issues` - Completed manifest issue numbers

## Step 2: Resolve Update Mode

For normal issue-linked runs, find the matching plan phase by `issue_number` as
before.

For `manifest_finalization=true`, resolve exactly one owning plan in this order:

1. Use `auto_mode_plan_id` when present and verify that exact plan exists.
2. Otherwise parse only a canonical trailing plan token from `source_branch`,
   such as `accumulate/E37-M1` -> `E37-M1`, and verify it exists.
3. If no canonical plan ID can be derived, fail closed. Completed issue coverage
   may verify ownership but must never select a plan by itself.

Before mutation, require every non-null phase issue number in the selected plan
to be present in `auto_mode_completed_issues` or already Shipped. Ambiguous,
partial, or unrepresented phase sets fail closed with `PLAN_UPDATE_SHORT_FAILED`;
never update multiple unrelated plans or mark an unexecuted phase Shipped.
Every unshipped phase must have a non-null issue number represented in
`auto_mode_completed_issues`; an issue-less unshipped phase has no completion
evidence and therefore fails closed.

For epic plans, do not use the phase-only promotion rule. Require every declared
child plan to already be Shipped/completed before promoting the epic; otherwise
return `PLAN_UPDATE_SHORT_FAILED` without mutation.

## Step 3: Find Matching Plan Phase

List active plans and scan phases for a matching `issue_number`:

```python
adw_plans_read({"command": "list", "lifecycle": "active", "options": "json"})
```

For each plan, check its `phases` array for an entry where
`phase.issue_number == issue_number`.

If no active match, this issue may not be tracked in a plan. Report
completion with no changes.

## Step 4: Mark Phase Or Plan Phases Shipped

```python
get_datetime({"format": "date"})

adw_plans_mutate({
  "command": "update-phase",
  "plan_id": "{plan_id}",
  "phase_id": "{phase_id}",
  "options": "phase-status=Shipped",
  "cwd": "{worktree_path}"
})
```

In manifest-finalization mode, invoke `update-phase` for every phase in the
resolved plan that is not already Shipped. Re-read the plan after mutations and
verify every phase is Shipped.

## Step 5: Check Plan Promotion

After marking the phase, re-read the plan to check if all phases are now
Shipped:

```python
adw_plans_read({"command": "show", "plan_id": "{plan_id}", "options": "json", "cwd": "{worktree_path}"})
```

If every phase has `status: "Shipped"`:

```python
adw_plans_mutate({
  "command": "update",
  "plan_id": "{plan_id}",
  "options": "status=Shipped",
  "cwd": "{worktree_path}"
})
```

If some phases remain, no plan-level promotion.

## Step 6: Report Completion

In manifest-finalization mode, write the terminal result to the finalizer's
message stream before returning:

```python
adw_spec_messages({
  "command": "messages-write",
  "adw_id": adw_id,
  "agent": "plan-update-short",
  "message": "PLAN_UPDATE_SHORT_COMPLETE plan=<plan_id> phases=<count> status=Shipped"
})
```

Write the corresponding bounded `PLAN_UPDATE_SHORT_FAILED` message on failure.

### Phase Shipped:

```
PLAN_UPDATE_SHORT_COMPLETE

Issue: #{issue_number}
Plan: {plan_id} ({plan_title})
Phase: {phase_id} - {phase_title}
Phase status: Shipped

Plan promotion: {Yes, all phases shipped / No, {n} phases remaining}
```

### No Matching Plan:

```
PLAN_UPDATE_SHORT_COMPLETE

Issue: #{issue_number}
No matching plan phase found. No updates needed.
```

Manifest finalization never treats a missing plan as a successful no-op. It
returns `PLAN_UPDATE_SHORT_FAILED` because finalization explicitly owns plan
closeout.

### Failure Case:

```
PLAN_UPDATE_SHORT_FAILED: {reason}

Issue: #{issue_number}
Error: {specific_error}
```

# Quick Reference

**Output Signal:** `PLAN_UPDATE_SHORT_COMPLETE` or `PLAN_UPDATE_SHORT_FAILED`

**Scope:** Metadata-only mutations via `adw_plans_read` and `adw_plans_mutate` (no file edits)

**Operations:** `update-phase` (mark Shipped) + `update` (promote plan status)

**Trigger:** Runs during the shipping step of workflows

**Fast:** Typically 3-4 tool calls total
