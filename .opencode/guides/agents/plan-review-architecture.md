# Plan Review Architecture Agent - Usage Guide

**Version:** 1.0.0  
**Last Updated:** 2026-03-28

## Overview

`plan-review-architecture` is a primary planner review-stage agent. It reviews plan docs under
`adw-docs/dev-plans/`, expands thin architecture/dependency sections in place, and reports a
bounded summary for downstream pipeline visibility.

## Responsibilities

- Target section 4 (Architecture & Design) and section 6 (Dependencies / Integration Points).
- Expand thin or ambiguous content while keeping scope aligned to the source plan.
- Preserve deterministic, in-place edits (no file relocation or structural rewrites).
- Emit concise status summaries (revised, passed, concerns) via workflow messages.

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

## Output Signals

- Success: `PLAN_REVIEW_ARCHITECTURE_COMPLETE`
- Failure: `PLAN_REVIEW_ARCHITECTURE_FAILED`

## Pipeline Placement

- Runs after drafting and phase sizing in the E15 planner review flow.
- Complements `plan-phase-splitter` by focusing on architecture/dependency quality, not sizing.

## Source Definition

- Agent file: `.opencode/agent/plan-review-architecture.md`
- Quick reference: `AGENTS.md` (Plan Review Pipeline Agents section)
