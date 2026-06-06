# Plan Type Designer Agent - Usage Guide

## Overview

`plan-type-designer` is a primary planning agent that designs and scaffolds
dynamic plan-type contracts through a deterministic, confirmation-gated flow.
It is intended for contract design workflows where safety, explicit approvals,
and predictable terminal outcomes are required.

## Core Responsibilities

- Capture plan-type intent and purpose before proposing structure.
- Propose candidate sections and metadata constraints.
- Present a recommended structure and at least one alternative.
- Require explicit confirmation before any write operation.
- Generate contract/template artifacts after confirmation.
- Validate outputs and emit deterministic terminal signals.

## Supported Intents

- `create-type`
- `clone-from-existing`
- `modify-type`

Inputs may arrive as natural language but should be normalized to:

- `intent`
- `target_type_name`
- `purpose`
- `required_sections` (optional)
- `metadata_constraints` (optional)

## Execution Contract

The agent follows explicit state transitions and confirmation gating.

### create-type flow

`intake -> intent_captured -> sections_drafted -> metadata_constrained -> options_presented -> awaiting_confirmation -> ready_to_write -> written -> validated`

### clone-from-existing flow

`intake -> source_selected -> delta_defined -> awaiting_confirmation -> ready_to_write -> written -> validated`

### modify-type flow

`intake -> target_selected -> patch_proposed -> awaiting_confirmation -> ready_to_write -> written -> validated`

## Guardrails

- No writes before explicit confirmation.
- Post-write validation is mandatory.
- On validation failure, stop and report rollback expectations.
- Do not reference unsupported or unimplemented runtime commands.

## CLI Parity Notes (create/clone/modify)

- `adw plans create-type` supports direct creation and clone-based creation via
  `--clone-from <existing-type>`.
- Non-interactive mode is fail-closed and requires **both** flags:
  `--non-interactive --yes`.
- `adw plans modify-type <type-name>` supports section and field mutations
  (`--add-section`, `--remove-section`, `--reorder-sections`, `--add-field`,
  `--remove-field`) and uses deterministic validation/error contracts.
- All commands support `--cwd <path>` for worktree-scoped operations.

## Integration Coverage and Verification

Integration coverage for plan-type CLI parity is implemented in:

- `adw/plans/tests/type_designer_integration_test.py`

Covered flows include:

- `create-type` (non-interactive create and scaffolding assertions)
- `clone-from-existing` (template/config copy assertions)
- `modify-type` (section/field mutation + persistence assertions)
- negative/contract paths (flag pairing and fail-closed error behavior)

Run targeted verification with:

```bash
pytest adw/plans/tests/type_designer_integration_test.py -v
```

## Implementation Notes (Helper Primitives)

The planner stack now includes shared helper primitives in
`adw/plans/type_designer.py` for deterministic, testable plan-type mutation
behavior. These helpers are intended for internal workflow/runtime use and
contributor extension work:

- `generate_plan_type_config(...)`: builds canonical candidate payloads with
  stable key ordering and defaults.
- `validate_type_config(...)`: validates candidate entries through
  `PlanTypeConfig` and enforces fail-closed duplicate `id_prefix` detection.
- `scaffold_type_templates(...)`: scaffolds deterministic section template files
  under `plans/templates/<plan-type>/`.
- `scaffold_instance_directories(...)`: creates
  `plans/<directory>/` and `plans/sections/<directory>/` with traversal-safe
  path confinement and `.gitkeep` markers.
- `read_plan_config(...)`: loads and shape-validates `plans/config.json`.
- `write_plan_type_entry(...)` / `update_plan_type_entry(...)`: perform
  validated, atomic config writes via temporary-file replacement.
- `analyze_modify_impact(...)`: summarizes planned `modify-type` impact before
  mutation writes (section/field deltas, affected plan counts, warnings).
- `add_section_to_type(...)` / `remove_section_from_type(...)` /
  `reorder_sections(...)`: apply deterministic section mutations with
  additive-only built-in guardrails and issue-template synchronization.
- `add_field_to_type(...)` / `remove_field_from_type(...)`: apply constrained
  metadata field updates with additive-only built-in guardrails.
- `_sync_issue_template_for_sections(...)`: reconciles
  `plans/templates/<plan-type>/issue_template.json` with section/role changes.
- `_archive_path(...)`: archives removed template assets to
  `plans/templates/_archive/` instead of hard-deleting them.
- `_mutate_sections_with_template_sync(...)`: wraps section mutation writes with
  issue-template sync and rollback-on-failure semantics.

These primitives do not change the external terminal-signal contract for
`plan-type-designer`; they standardize internal safety and persistence
semantics for create/modify flows, especially `modify-type` rollback and
archive behavior.

## Terminal Signals

- Success: `PLAN_TYPE_CREATED`
- Failure: `PLAN_TYPE_CREATION_FAILED`

Failure output must include deterministic:

- `stage`
- `reason`

## Tool Profile

Primary tools:

- `read`, `edit`, `write`, `list`, `ripgrep`, `move`
- `todoread`, `todowrite`
- `adw_spec`, `adw_plans`
- `feedback_log`, `get_datetime`

## Canonical Agent Contract

Source-of-truth definition:

- `.opencode/agent/plan-type-designer.md`

Related operator coverage:

- `docs/Examples/operations/planner-workflow-runbook.md`
