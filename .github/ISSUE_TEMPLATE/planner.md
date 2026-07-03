---
name: Planner Rough Scope Request
about: Structured planner intake template for deterministic scope analysis
title: "[Planner]: "
labels: ["agent", "blocked", "model:default", "type:planner"]
assignees: ""
---

## Type
- [ ] Feature planning
- [ ] Maintenance planning
- [ ] Research planning
- [ ] Multi-track epic planning

## Vision
- One-paragraph outcome statement:
- Intended users/owners:
- Business or technical value:

## Problem Statement
- Current state:
- Pain points:
- Why now:

## Rough Scope
| Area | In Scope | Out of Scope |
|------|----------|--------------|
| Functional | - [ ] | - [ ] |
| Technical | - [ ] | - [ ] |
| Validation | - [ ] | - [ ] |

## Codebase Research References
Use this section for concrete findings from `codebase-researcher` or manual repo review.

| Reference | What It Shows | Planning Impact |
|-----------|---------------|-----------------|
| `path/to/file.py:123` |  |  |
| `path/to/test_file.py:45` |  |  |
| `docs/path.md` |  |  |

## Examples / Prior Art
- Similar implementation patterns:
  - `path/to/example.py:10` --
- Similar tests or validation fixtures:
  - `path/to/example_test.py:20` --
- Similar docs, runbooks, or operator flows:
  - `docs/path.md` --
- Anti-patterns or approaches to avoid:
  - `path/to/legacy.py:30` --

## Child Tracks
Track Type must be one of `Feature`, `Maintenance`, or `Research`.
All child tracks in a planner request must use the same Track Type; do not mix
Feature, Maintenance, and Research child plans in one request.

| Track ID | Track Type | Goal | Size (S/M/L) | Dependencies |
|----------|------------|------|--------------|--------------|
| T1 |  |  |  |  |
| T2 |  |  |  |  |

## Suggested Phases By Track
List enough phase detail for each track that the planner can create concrete feature,
maintenance, or research phases instead of generic placeholders.

| Track ID | Suggested Phases | Validation / Done Signal |
|----------|------------------|--------------------------|
| T1 | 1.  |  |
| T2 | 1.  |  |

## Dependencies
- Internal dependencies:
  - [ ] None
  - [ ] Listed below
- External dependencies:
  - [ ] None
  - [ ] Listed below

## Related Plan
- Related Plan ID (optional):
- Allowed formats: E{n}, F{n}, E{n}-F{m}, M{n}, E{n}-M{m}, R{n}, E{n}-R{m}
<!-- ADW Agent: Use `adw_plans({ command: "list", json: true })` for discovery.
Use `adw_plans({ command: "show", plan_id: "<id>", json: true })` for verification. -->

## Constraints
- Timeline constraints:
- Resource constraints:
- Tooling/platform constraints:
- Compliance/security constraints:

## Success Metrics
- [ ] Clear acceptance criteria defined
- [ ] Risks and mitigations identified
- [ ] Child tracks are dependency-ordered
- [ ] Validation strategy is testable
- [ ] Codebase references and examples are detailed enough for planner handoff
- [ ] Each child track has an explicit type and suggested phases
