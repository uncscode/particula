# Feature: Rectangular Wall Loss Strategy

**Status:** In Progress
**Priority:** P2
**Assignees:** ADW Workflow
**Labels:** feature, dynamics, wall-loss
**Milestone:** v0.2.x
**Size:** M (~120 LOC core + tests)

**Start Date:** 2025-12-17
**Target Date:** 2025-12-31
**Created:** 2025-12-17
**Updated:** 2025-12-17

**Related Issues:** #817 (parent: #72)
**Related PRs:** (pending)
**Related ADRs:** [ADR-001][adr-001] (strategy-based wall loss subsystem)

---

## Overview

Extend the wall loss strategy system with a rectangular geometry strategy
that mirrors the existing spherical implementation. The new
`RectangularWallLossStrategy` supports all particle distribution types and is
exported via `particula.dynamics`, keeping the public API consistent across
geometries.

### Problem Statement

Spherical wall loss coverage leaves common box-shaped chambers unsupported.
Users need a strategy that handles rectangular dimensions without leaving the
strategy API or dropping down to standalone rate helpers.

### Value Proposition

- Enables simulations of rectangular/box chambers using the strategy API.
- Preserves distribution-type coverage (discrete, continuous_pdf,
  particle_resolved).
- Aligns with existing wall loss abstractions and exports for minimal user
  friction.

## Phases

> **Philosophy:** Each phase should be one GitHub issue with a focused scope
> (~100 lines of code or less, excluding tests/docs).

- [x] **Phase 1:** Add rectangular strategy, exports, and mirrored tests
  - GitHub Issue: #817
  - Status: Completed
  - Size: M (~120 LOC core + tests)
  - Dependency: Spherical strategy baseline (#816) and wall loss helpers
  - Estimated Effort: 2–3 days

## User Stories

### Story 1: Rectangular wall loss for chamber simulations
**As a** chamber simulation developer
**I want** a rectangular wall loss strategy that matches the strategy API
**So that** I can model box-shaped chambers without custom glue code

**Acceptance Criteria:**
- [ ] Strategy validates three positive chamber dimensions
- [ ] Strategy supports discrete, continuous_pdf, particle_resolved
- [ ] Loss rates remain finite across typical aspect ratios
- [ ] Exports available via `particula.dynamics`

## Technical Approach

### Architecture Changes

Introduce `RectangularWallLossStrategy` that subclasses `WallLossStrategy` and
reuses `get_rectangle_wall_loss_coefficient_via_system_state`.

**Affected Components:**
- `particula.dynamics.wall_loss.wall_loss_strategies`: new rectangular strategy
- `particula.dynamics.wall_loss.__init__`: export strategy
- `particula.dynamics.__init__`: re-export strategy for public API
- Tests under `particula/dynamics/wall_loss/tests` and `particula/dynamics/tests`

### Design Patterns

- Strategy pattern via `WallLossStrategy` ABC.
- Input validation (`@validate_inputs`) for eddy diffusivity and explicit
  dimension checks (length == 3, all positive).
- Vectorized coefficient computation to avoid per-particle loops.

### API Surface

- `particula.dynamics.wall_loss.RectangularWallLossStrategy`
- `particula.dynamics.RectangularWallLossStrategy` (public export)
- Methods: `loss_coefficient`, `rate`, `loss_rate`, `step`

## Implementation Tasks

- [x] Implement `RectangularWallLossStrategy` with dimension validation
- [x] Wire exports in `wall_loss/__init__.py` and `dynamics/__init__.py`
- [x] Mirror tests for rectangular geometry in wall_loss scope
- [x] Mirror smoke and validation tests in dynamics scope
- [ ] Update broader docs/examples if needed after review

## Testing Strategy

### Unit Tests

- Wall-loss scope (`particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py`):
  validates dimension length/positivity, eddy diffusivity guard, distribution
  type coverage (discrete, continuous_pdf, particle_resolved), aspect ratios
  (cubic, elongated, flat), parity with `get_rectangle_wall_loss_rate`,
  particle-resolved mask lengths, zero-concentration stability, and small/large
  particle robustness.
- Dynamics scope (`particula/dynamics/tests/wall_loss_strategies_test.py`):
  mirrors coverage to confirm public exports, bad distribution handling, and
  finite outputs via the `particula.dynamics` import path.

### Integration/Behavior

- `rate` and `step` verified to reduce concentration for discrete and
  continuous_pdf distributions and to handle empty particle-resolved inputs
  without shape errors.

## Documentation

- [x] Feature documentation (this file)
- [ ] API documentation updates (if new reference pages needed)
- [ ] User guide updates (examples/notebooks)

## Success Criteria

- [ ] Rectangular strategy uses rectangle coefficient helper and passes
      validation
- [ ] Supports discrete, continuous_pdf, particle_resolved in rate/step paths
- [ ] Exports present in `particula.dynamics` and wall_loss package
- [ ] Mirrored tests in wall_loss and dynamics scopes pass
- [ ] Examples and docs refreshed where necessary

## Usage Example

```python
import particula as par

strategy = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1.0e-4,
    chamber_dimensions=(1.0, 0.5, 0.5),
    distribution_type="discrete",
)

rate = strategy.rate(
    particle=particle,
    temperature=298.0,
    pressure=101325.0,
)

particle = strategy.step(
    particle=particle,
    temperature=298.0,
    pressure=101325.0,
    time_step=1.0,
)
```

## Change Log

| Date       | Change                                | Author       |
|------------|---------------------------------------|--------------|
| 2025-12-17 | Initial feature documentation created | ADW Workflow |

[adr-001]: ../../architecture/decisions/ADR-001-strategy-based-wall-loss-subsystem.md
