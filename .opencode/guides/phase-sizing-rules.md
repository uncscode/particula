# Phase Sizing Rules (Canonical)

This document is the canonical source for planner phase-sizing and split behavior.

## Split Policy

- `XS`: no split.
- `S`: no split.
- `M`: optional split into **2** S-sized phases only when concerns are clearly independent.
- `L`: mandatory split into **3-5** S-sized sub-phases.
- `XL`: mandatory split into **5+** S-sized sub-phases.

## Parsing Contract

- Supported phase ID variants: `(?:E\d+-F\d+|F\d+|E\d+|M\d+)-P\d+`
- Supported checkbox markers: `- \[(?: |x|X)\]`
- `Size:` token is case-normalized to uppercase before decisions.

## Fallback Behavior

- Missing `Size:` defaults to `S` and increments a warning counter.
- Malformed phase IDs are warning-only and skipped (non-fatal).
- Maintenance phases that rely on worktree `--cwd` flows must include
  path-resolution and rerun-verification effort in sizing estimates.

## Consumers

- `.opencode/agent/plan-phase-splitter.md`
- `adw-docs/agents/plan-phase-splitter.md`
- `plans/features/E15-F4.json`
- `adw/tests/plan_phase_splitter_agent_test.py`
