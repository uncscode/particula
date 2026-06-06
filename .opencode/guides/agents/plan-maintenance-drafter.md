# Plan Maintenance Drafter Agent - Usage Guide

**Version:** 1.0.0  
**Last Updated:** 2026-03-26

## Overview

`plan-maintenance-drafter` is a subagent that produces first-pass maintenance plan documents for
the plan-document pipeline. It reads the live maintenance template at runtime, incorporates
workflow context, and writes deterministic output under `plans/sections/maintenance/`.

Unlike epic/feature drafters, this agent intentionally does not delegate to
`codebase-researcher` and keeps `task: false`.

## Responsibilities

- Parse orchestrator assignment context (`adw_id`, `target_id`, optional `parent_id`).
- Handle both epic-linked and standalone maintenance scenarios.
- Read `plans/templates/maintenance/` at runtime for drift resilience.
- Read prior workflow messages via `adw_spec messages-read`.
- Draft all required maintenance sections with first-pass content.
- Add phases via `adw_plans add-phase` and scaffold sections via
  `adw_plans scaffold-sections`.
- Write deterministic output to canonical section files under
  `plans/sections/maintenance/{MAINT_ID}/`.
- Report draft completion and challenges via `adw_spec messages-write`.

## Drafting Contract

The drafter must populate all 11 canonical maintenance sections. The canonical keys must match
`adw/plans/sections.py` (`MaintenanceSections`) exactly:

- `purpose_justification`
- `scope`
- `guidelines_requirements`
- `success_criteria`
- `phase_details`
- `testing_requirements`
- `example_tasks`
- `dependencies`
- `communication_reporting`
- `open_questions`
- `change_log`

Canonical maintenance section paths follow:

- `plans/sections/maintenance/<maintenance-id>/<section>.md`
- `<maintenance-id>`: `M{n}` or `E{n}-M{m}`
- `<section>`: lowercase `snake_case` and one of the canonical keys above

This docs contract is intentionally kept in parity with repository enforcement in
`adw/plans/repository.py` (canonical maintenance path + configured key membership checks).

When context is missing, continue with template-complete placeholders and record gaps in the
completion `challenges` list.

## Step 7 Completion Payload Contract

Completion messages must include payload keys in this exact order:

1. `scenario`
2. `sections_drafted`
3. `challenges`

Allowed `scenario` values:

- `standalone-maintenance`
- `epic-linked-maintenance`

Example shape:

```text
{
  "scenario": "standalone-maintenance",
  "sections_drafted": 11,
  "challenges": []
}
```

## Failure and Fallback Behavior

- Build full draft content before replacing any existing target file.
- If path safety validation fails or write/replace fails, emit the failure signal and avoid
  partial target state.
- If assignment context is sparse, continue with template-complete placeholders and include
  limitations in the completion message.

## Enabled Tools (Least Privilege)

- `read`
- `write`
- `edit`
- `move`
- `list`
- `ripgrep`
- `adw_spec`
- `adw_plans`
- `feedback_log`
- `todoread`
- `todowrite`
- `get_datetime`

## Output Signals

- Success: `PLAN_MAINTENANCE_DRAFTER_COMPLETE`
- Failure: `PLAN_MAINTENANCE_DRAFTER_FAILED`

Failure messages should include cause and actionable rerun guidance.

## Related Agents

- Parent dispatcher: `plan-orchestrator`
- Sibling drafters: `plan-epic-drafter`, `plan-feature-drafter`

## Source Definition

- Agent file: `.opencode/agent/plan-maintenance-drafter.md`
- Quick reference: `AGENTS.md` (Plan Document Pipeline Agents section)
- Orchestration guide: `adw-docs/agents/plan-orchestrator.md`
