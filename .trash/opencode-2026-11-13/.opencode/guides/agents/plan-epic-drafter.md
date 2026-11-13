# Plan Epic Drafter Agent - Usage Guide

**Version:** 1.0.0  
**Last Updated:** 2026-03-26

## Overview

`plan-epic-drafter` is a subagent that produces first-pass epic plan documents for the
plan-document pipeline. It reads the live template at runtime, incorporates workflow context,
uses `codebase-researcher` for technical enrichment, and writes deterministic output under the
epics plan directory.

## Responsibilities

- Parse orchestrator assignment context (`adw_id`, `epic_id`, feature tracks, maintenance tracks).
- Read `adw-docs/dev-plans/template-epic.md` at runtime for drift resilience.
- Read prior workflow messages via `adw_spec messages-read`.
- Invoke `codebase-researcher` for architecture/module context.
- Draft all required epic sections with first-pass content.
- Write deterministic output to
  `adw-docs/dev-plans/epics/{EPIC_ID}-{slug}.md`.
- Report draft completion, thin sections, and challenges via `adw_spec messages-write`.

## Drafting Contract

The drafter must populate the full epic template section structure, including:

- Vision and outcomes
- Scope and child tracks
- Dependencies, phases, milestones, and implementation strategy
- Governance, risks, and success metrics
- Rollout, validation, and appendix content

When feature or maintenance track lists are empty, the generated document should still include
those sections with explicit `none/empty` placeholders.

## Failure and Fallback Behavior

- If `codebase-researcher` fails, times out, or returns sparse output, continue drafting using
  orchestrator context + workflow messages + template structure.
- Build full draft content before replacing any existing target file.
- If write/replace fails, emit a failure signal and avoid leaving a partial target state.

## Enabled Tools (Least Privilege)

- `read`
- `write`
- `edit`
- `move`
- `list`
- `ripgrep`
- `task`
- `adw_spec`
- `todoread`
- `todowrite`
- `get_datetime`

## Output Signals

- Success: `PLAN_EPIC_DRAFTER_COMPLETE`
- Failure: `PLAN_EPIC_DRAFTER_FAILED`

Failure messages should include cause and actionable rerun guidance.

## Related Agents

- Parent dispatcher: `plan-orchestrator`
- Context provider: `codebase-researcher`
- Sibling drafters: `plan-feature-drafter`, `plan-maintenance-drafter`

## Source Definition

- Agent file: `.opencode/agent/plan-epic-drafter.md`
- Quick reference: `AGENTS.md` (Plan Document Pipeline Agents section)
- Orchestration guide: `adw-docs/agents/plan-orchestrator.md`
