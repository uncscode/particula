# Plan Phase Splitter Agent - Usage Guide

**Version:** 1.0.0  
**Last Updated:** 2026-03-27

## Overview

`plan-phase-splitter` is a primary planner pipeline agent that enforces phase-sizing policy
across canonical section files in `plans/sections/`. It scans scoped plan markdown, parses
checklist phase metadata (`Size:`), and applies deterministic split rules before downstream
review stages.

Canonical source for split behavior: `adw-docs/phase-sizing-rules.md`.

## Responsibilities

- Discover candidate plan docs once using `adw_plans list-sections` and scoped files under
  `plans/sections/{epics,features,maintenance}/`.
- Early-filter files that do not contain checklist/phase markers before full parsing.
- Read each candidate file once and parse phase IDs plus `Size:` metadata in one pass.
- Apply split rules for `XS`, `S`, `M`, `L`, and `XL` sizes.
- Ensure any newly created sub-phase includes a `tests-with-feature` annotation.
- Write bounded aggregate summary output via `adw_spec messages-write`.

## Split Rules

See `adw-docs/phase-sizing-rules.md` for the canonical thresholds and parsing
contract. Summary:

- `XS`: no split
- `S`: no split
- `M`: optional split into two S-sized phases only when independent concerns are clear
- `L`: required split into 3-5 S-sized sub-phases
- `XL`: required split into 5+ S-sized sub-phases

Fallback and warning behavior:

- Missing `Size:` defaults to `S` and increments warning counters.
- Malformed phase IDs are warning-only and skipped (non-fatal).
- No matching checklist entries results in a clean no-op summary.

Discovery excludes templates/indexes/completed/archive artifacts.

## Enabled Tools (Least Privilege)

- `read`
- `write`
- `edit`
- `move`
- `list`
- `ripgrep`
- `adw_spec`
- `todoread`
- `todowrite`
- `get_datetime`
- `feedback_log`

## Output Signals

- Success: `PLAN_PHASE_SPLITTER_COMPLETE`
- Failure: `PLAN_PHASE_SPLITTER_FAILED`

## Pipeline Placement

- Runs after plan drafting and before review agents in the planner flow.
- Publishes bounded counters and split summaries to workflow messages for downstream stages.

## Source Definition

- Agent file: `.opencode/agent/plan-phase-splitter.md`
- Quick reference: `AGENTS.md` (Plan Document Pipeline Agents section)
