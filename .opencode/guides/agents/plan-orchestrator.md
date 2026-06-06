# Plan Orchestrator Agent - Usage Guide

**Version:** 1.0.0  
**Last Updated:** 2026-03-26

## Overview

`plan-orchestrator` is a primary agent that coordinates plan-document drafting after scope
classification. It reads workflow messages from `plan-scope-analyzer`, selects the latest valid
classification payload, creates plan records with required `cwd: "<worktree_path>"`, builds a
deterministic dispatch plan, and calls drafter subagents in order.

For every `adw_plans create` call, resolve `worktree_path` from ADW state first and pass that
exact absolute path as `cwd`; do not hardcode repo-relative worktree guesses.

## Responsibilities

- Read classifier output from `adw_spec messages-read` using bounded windows.
- Select the most recent `plan-scope-analyzer` message.
- Parse `plan_type`, `epic_id`, and feature/research/maintenance track lists.
- Create epic, feature, research, and maintenance plan records via `adw_plans create`.
- Build one in-memory dispatch list.
- Dispatch drafters sequentially with best-effort continuation on partial failures.
- Write a completion/error summary via `adw_spec messages-write`, preserving
  `review_plan_ids` as the downstream handoff field.

Minimal operator example:

```bash
uv run adw spec read --adw-id <id> --field worktree_path --raw
uv run adw plans create --type research --title "<title>" --cwd <worktree_path>
```

## Dispatch Contract

The orchestrator dispatches subagents in this deterministic order:

- `plan_type=epic`
  1. `plan-epic-drafter` for `epic_id`
  2. `plan-feature-drafter` for each feature track (listed order)
  3. `plan-research-drafter` for each research track (listed order)
  4. `plan-maintenance-drafter` for each maintenance track (listed order)
- `plan_type=feature`
  - One `plan-feature-drafter` call using the first normalized `feature_tracks` ID
- `plan_type=research`
  - One `plan-research-drafter` call using the first normalized `research_tracks` ID
- `plan_type=maintenance`
  - One `plan-maintenance-drafter` call using the first normalized `maintenance_tracks` ID

Canonical default order is `epic -> feature -> research -> maintenance`. Keep
the dependency-analysis override note narrow: only surface a research-first
exception when the issue or classifier context explicitly requires it.

If `plan_type=unknown`, classifier data is missing, or required IDs are invalid, the agent exits
before dispatch and writes an actionable error summary.

Safety guardrails:
- Provenance validation: selected message must be authored by `plan-scope-analyzer` and match
  current workflow identifiers when `workflow_id`/`adw_id`/`run_id` are present.
- ID sanitization before prompt interpolation.
- Epic dispatch fan-out cap (`MAX_TRACK_DISPATCH = 25`).

## Enabled Tools (Least Privilege)

- `task`
- `adw_spec`
- `read`
- `list`
- `ripgrep`
- `todoread`
- `todowrite`
- `get_datetime`

## Output Signals

- Success: `PLAN_ORCHESTRATOR_COMPLETE`
- Pre-dispatch fatal error: `PLAN_ORCHESTRATOR_FAILED`

## Related Agents

- Upstream: `plan-scope-analyzer`
- Downstream:
  - `plan-epic-drafter`
  - `plan-feature-drafter`
  - `plan-research-drafter`
  - `plan-maintenance-drafter`

## Source Definition

- Agent file: `.opencode/agent/plan-orchestrator.md`
- Quick reference: `AGENTS.md` (Plan Document Pipeline Agents section)
