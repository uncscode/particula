# Feature: WallLoss Runnable Process

**Status:** Completed
**Priority:** P1
**Assignees:** ADW Workflow
**Labels:** feature, dynamics, wall-loss
**Milestone:** v0.2.x
**Size:** M (~150 LOC core + tests)

**Start Date:** 2025-12-17
**Target Date:** 2025-12-18
**Created:** 2025-12-18
**Updated:** 2025-12-18

**Related Issues:** #819 (parent: #72; deps: #816, #817, #818)
**Related PRs:** (pending)
**Related ADRs:** [ADR-001][adr-001] (strategy-based wall loss subsystem)

---

## Overview

Add a `WallLoss` runnable that wraps existing wall loss strategies and plugs
into the `RunnableABC` pipeline for aerosol processes. The runnable splits the
provided `time_step` across `sub_steps`, delegates `rate` and `step` to the
strategy, clamps concentrations to non-negative after each sub-step, and
exports via `particula.dynamics.WallLoss` alongside existing wall loss API
surfaces.

### Problem Statement

Wall loss strategies existed but lacked a runnable that integrates with the
`RunnableABC` pattern and process chaining. Users had to call strategies
manually, losing consistency with coagulation and condensation runnables.

### Value Proposition

- Enables wall loss to participate in runnable chains (`|`) for aerosols.
- Preserves strategy polymorphism while enforcing non-negative concentrations.
- Keeps public API coherent: `particula.dynamics.WallLoss` mirrors other
  runnables.

## Phases

> **Philosophy:** Each phase should be one GitHub issue with a focused scope
> (~100 lines of code or less, excluding tests/docs).

- [x] **Phase 1:** Implement runnable + exports + runnable tests
  - GitHub Issue: #819 (parent: #72)
  - Status: Completed
  - Size: M (~150 LOC core + tests)
  - Dependencies: #816 (strategy ABC), #817 (rectangular strategy),
    #818 (builders/factory)
  - Acceptance: Runnable implements RunnableABC, splits sub-steps,
    clamps non-negative, exports via `particula.dynamics`, tests green

## User Stories

### Story 1: Chainable wall loss process
**As a** dynamics developer
**I want** a wall loss runnable that plugs into `RunnableABC`
**So that** I can chain wall loss with coagulation and condensation without
custom glue code

**Acceptance Criteria:**
- [ ] Runnable delegates `rate` and `step` to provided wall loss
      strategy
- [ ] `execute` splits `time_step` across `sub_steps` and clamps
      concentrations
- [ ] Supports discrete and particle_resolved distribution types
- [ ] Exported via `particula.dynamics.WallLoss`
- [ ] Runnable-level tests cover sub-steps, clamp, chaining, and strategy
      types

## Technical Approach

### Architecture Changes

`WallLoss` runnable added to `particula/dynamics/particle_process.py`, following
existing runnable patterns. The runnable stores a `WallLossStrategy`, delegates
`rate`/`step`, splits `time_step` by `sub_steps`, clamps concentrations to
non-negative after each sub-step, returns the updated aerosol, and is
exported next to wall loss strategies, builders, and factory.

**Affected Components:**
- `particula.dynamics.particle_process.WallLoss` (new runnable)
- `particula.dynamics.__init__` (public export)
- `particula.dynamics.tests.wall_loss_runnable_test` (new tests)

### Design Patterns

- Runnable pattern via `RunnableABC` with process chaining (`|`).
- Strategy delegation for `rate` and `step` to wall loss strategies.
- Safety clamp to keep concentrations non-negative after each sub-step.

## Implementation Tasks

- [x] Add `WallLoss` runnable that splits `time_step` over `sub_steps`
- [x] Delegate `rate`/`step` to provided wall loss strategy
- [x] Clamp concentrations to non-negative after each sub-step
- [x] Export runnable via `particula.dynamics.__init__`
- [x] Add runnable tests (init, execute, rate, clamp, sub-steps,
      chaining; discrete + particle_resolved; spherical + rectangular)

## Testing Strategy

### Unit / Behavior Tests

- `particula/dynamics/tests/wall_loss_runnable_test.py` covers:
  - Runnable init with spherical and rectangular strategies
  - `execute` reduces concentration and respects zero `time_step`
  - Non-negative clamp after large time steps
  - `sub_steps` call count and time-split behavior
  - `rate` shape/sign checks
  - Distribution types: discrete, particle_resolved
  - Chaining with coagulation and condensation runnables

### Integration

- Import smoke via `particula.dynamics.WallLoss` to ensure public export and
  chaining compatibility.

## Documentation

- [x] Feature documentation (this file)
- [ ] API documentation updates (if reference pages needed)
- [ ] User guide updates (examples/notebooks)

## Success Criteria

- [ ] `WallLoss` inherits `RunnableABC` and delegates to wall loss
      strategies
- [ ] `execute` splits `time_step` across `sub_steps` and clamps
      concentrations
- [ ] Supports discrete and particle_resolved distributions
- [ ] Export available via `particula.dynamics.WallLoss`
- [ ] Runnable tests pass (init, execute, rate, clamp, chaining)

## Change Log

| Date       | Change                                | Author       |
|------------|---------------------------------------|--------------|
| 2025-12-18 | Initial feature documentation created | ADW Workflow |

[adr-001]: ../../architecture/decisions/ADR-001-strategy-based-wall-loss-subsystem.md
