# Plan Research Drafter Agent - Usage Guide

**Version:** 1.0.0  
**Last Updated:** 2026-06-02

## Overview

`plan-research-drafter` is a live subagent for first-pass research plan drafting in the
plan-document pipeline. `plan-orchestrator` creates research plans, dispatches this drafter in the
canonical default order `epic -> feature -> research -> maintenance`, and preserves any explicit
dependency-driven research-first exception as a narrow override note. This agent reads live
research templates at runtime, uses `codebase-researcher` for targeted technical enrichment, and
writes deterministic output under the canonical research section directory.

## Responsibilities

- Parse orchestrator assignment context (`adw_id`, `target_id`, `plan_type=research`, optional
  `parent_id`).
- Handle both epic-linked and standalone research scenarios.
- Read `.opencode/plans/templates/research/` at runtime for drift resilience.
- Read prior workflow messages via `adw_spec_messages messages-read`.
- Resolve `worktree_path` before any worktree-scoped `adw_plans_read` or
  `adw_plans_mutate` call.
- Invoke `codebase-researcher` only for scoped architecture, module, data-source, and evaluation
  context.
- Draft all required research sections with first-pass content.
- Write deterministic output to `.opencode/plans/sections/research/{RESEARCH_ID}/`.
- Report draft completion, reduced-context notes, and challenges via `adw_spec_messages`.

## Drafting Contract

The drafter must populate all canonical research sections from the runtime template/config source:

- `overview`
- `scope`
- `data_sources`
- `methodology`
- `evaluation_strategy`
- `dependencies`
- `success_criteria`
- `risk_register`
- `open_questions`
- `change_log`

Canonical research section paths follow:

- `.opencode/plans/sections/research/{RESEARCH_ID}/<section>.md`
- `{RESEARCH_ID}`: `R{n}` or `E{n}-R{m}`
- `<section>`: lowercase `snake_case` and one of the canonical keys above

When context is missing, continue with template-complete placeholders and record the gap in
`reduced_context_notes` or `challenges`.

## Worktree and Path Safety

- Read `worktree_path` from `adw_spec_read` before any `adw_plans_read` or
  `adw_plans_mutate` call that needs `cwd`.
- Use `cwd: "<worktree_path>"` from repo root or `cwd: "."` only when already in the target
  worktree.
- Never use nested relative paths such as `./trees/{adw_id}`.
- Reject absolute-path escapes, traversal, and symlink escape paths before writing section files.
- Treat `adw_plans_read list-sections` output as advisory until it is confirmed to remain under
  `.opencode/plans/sections/research/{RESEARCH_ID}/`.

## Enabled Tools (Least Privilege)

- `read`
- `write`
- `edit`
- `list`
- `ripgrep`
- `task` (for `codebase-researcher` delegation only)
- `adw_spec_read`
- `adw_spec_messages`
- `adw_plans_read`
- `adw_plans_mutate`
- `feedback_log`
- `todowrite`
- `get_datetime`

The task surface is mechanically restricted by frontmatter `subagent_type_allowlist` to
`codebase-researcher` only.

## Output Signals

- Success: `PLAN_RESEARCH_DRAFTER_COMPLETE`
- Failure: `PLAN_RESEARCH_DRAFTER_FAILED`

Failure messages should include the cause, whether reduced-context drafting was attempted, and
actionable rerun guidance.

## Related Agents

- Dispatcher integration: `plan-orchestrator`
- Context provider: `codebase-researcher`
- Sibling drafters: `plan-epic-drafter`, `plan-feature-drafter`, `plan-maintenance-drafter`

## Source Definition

- Agent file: `.opencode/agent/plan-research-drafter.md`
- Quick reference: `AGENTS.md` (Plan Document Pipeline Agents section)
- Agent index: `.opencode/guides/agents/README.md`
