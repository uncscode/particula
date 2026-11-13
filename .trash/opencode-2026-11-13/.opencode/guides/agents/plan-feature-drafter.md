# Plan Feature Drafter Agent - Usage Guide

**Version:** 1.0.0  
**Last Updated:** 2026-03-26

## Overview

`plan-feature-drafter` is a subagent that produces first-pass feature plan documents for the
plan-document pipeline. It reads the live feature template at runtime, incorporates workflow
context, uses `codebase-researcher` for technical enrichment, and writes deterministic output
under the feature plan directory.

## Responsibilities

- Parse orchestrator assignment context (`adw_id`, `feature_id`, `feature_title`, scope notes).
- Handle both epic-linked and standalone feature scenarios.
- Read `adw-docs/dev-plans/template-feature.md` at runtime for drift resilience.
- Read prior workflow messages via `adw_spec messages-read`.
- Invoke `codebase-researcher` for architecture/module context.
- Draft all required feature sections with first-pass content.
- Write deterministic output to
  `adw-docs/dev-plans/features/{FEATURE_ID}-{slug}.md`.
- Report draft completion and challenges via `adw_spec messages-write`.

## Drafting Contract

The drafter must populate all feature template sections, including:

- Overview and scope framing
- Phase checklist and implementation tasks
- Architecture, dependencies, and testing strategy
- Documentation updates, success metrics, rollout, open questions, and change log

When context is missing, the draft should continue with explicit placeholders and record gaps in
the challenges summary.

## Failure and Fallback Behavior

- If `codebase-researcher` fails, times out, or returns sparse output, continue drafting using
  orchestrator context + workflow messages + template structure.
- Build full draft content before replacing any existing target file.
- If path safety validation fails or write/replace fails, emit the failure signal and avoid
  partial target state.

## Enabled Tools (Least Privilege)

- `read`
- `write`
- `edit`
- `list`
- `ripgrep`
- `task` (for `codebase-researcher` delegation only)
- `adw_spec`
- `todoread`
- `todowrite`
- `get_datetime`

## Output Signals

- Success: `PLAN_FEATURE_DRAFTER_COMPLETE`
- Failure: `PLAN_FEATURE_DRAFTER_FAILED`

Failure messages should include cause and actionable rerun guidance.

## Related Agents

- Parent dispatcher: `plan-orchestrator`
- Context provider: `codebase-researcher`
- Sibling drafters: `plan-epic-drafter`, `plan-maintenance-drafter`

## Source Definition

- Agent file: `.opencode/agent/plan-feature-drafter.md`
- Quick reference: `AGENTS.md` (Plan Document Pipeline Agents section)
- Orchestration guide: `adw-docs/agents/plan-orchestrator.md`
