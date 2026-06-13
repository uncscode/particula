# Feature: Charged/Electrostatic Wall Loss Strategy

**Status:** In Progress
**Priority:** P1
**Assignees:** ADW Workflow
**Labels:** feature, dynamics, wall-loss, charged
**Milestone:** v0.2.x
**Size:** L (~250 LOC core + tests/docs)

**Start Date:** 2025-12-21
**Target Date:** 2026-01-09
**Created:** 2025-12-21
**Updated:** 2025-12-21

**Related Issues:** [#821](https://github.com/uncscode/particula/issues/821) (parent: [#72](https://github.com/uncscode/particula/issues/72))
**Related PRs:** (pending)
**Related ADRs:** [ADR-001][adr-001]

---

## Overview

Add a charged/electrostatic wall loss strategy that augments the neutral wall
loss model with image-charge effects, optional electric-field drift, and
charge-dependent diffusion enhancement across spherical and rectangular
geometries. The strategy integrates with builders, factory, and exports to keep
wall loss configuration ergonomic and consistent with other dynamics modules.

### Problem Statement

Current wall loss support is neutral-only. Charged particles in chamber
simulations experience electrostatic attraction/repulsion and field-driven
drift, which the neutral model cannot capture. Without a charged-aware
strategy, charged particle simulations under-estimate deposition and miss
sign-dependent behavior.

### Value Proposition

- More accurate wall loss for charged particles, improving chamber fidelity.
- Sign-aware behavior: positive/negative charge effects and optional E-field
  drift captured without custom glue code.
- Builder/factory ergonomics with validated geometry selection and new charged
  parameters (`wall_potential`, `wall_electric_field`).
- Backward compatibility: reduces to neutral when charge and field are zero
  while still honoring image-charge effects when wall_potential is zero.

## Phases

> **Philosophy:** Each phase should be one GitHub issue with a focused scope
> (~100 lines of code or less, excluding tests/docs).

- [ ] **Phase 1:** Implement charged strategy, mixins, builder/factory wiring,
      tests, and docstrings
  - GitHub Issue: #821
  - Status: In Progress
  - Size: L (~250 LOC core + tests/docs)
  - Dependencies: #816 (spherical), #817 (rectangular), #818 (builders/factory),
    #819 (runnable)
- [ ] **Phase 2:** Update wall loss examples and user docs with charged usage
  - GitHub Issue: #821 follow-up
  - Status: Planned
  - Size: S (~60 LOC docs/examples)

## User Stories

### Story 1: Charged chamber simulations
**As a** chamber simulation developer
**I want** wall loss that accounts for particle charge and electric fields
**So that** deposition reflects sign-dependent attraction/repulsion and drift

**Acceptance Criteria:**
- [ ] Charged strategy differs from neutral for non-zero charge
- [ ] Image-charge effect active even when `wall_potential == 0`
- [ ] Electric-field drift optional and sign-aware; zero field reduces to
      neutral+image-charge only
- [ ] Reduces to neutral when charge and field are zero

### Story 2: Ergonomic configuration via builder/factory
**As a** user configuring wall loss
**I want** charged parameters available through builders and factory
**So that** I can pick geometry and charged options without manual wiring

**Acceptance Criteria:**
- [ ] Builder validates geometry (radius vs dimensions) and charged parameters
- [ ] Factory returns charged strategy via `strategy_type="charged"`
- [ ] Exports available under `particula.dynamics` and `wall_loss`
- [ ] Particle-resolved mode keeps survival probabilities within [0, 1]

## Technical Approach

### Architecture Changes

- Add `ChargedWallLossStrategy` subclass of `WallLossStrategy` handling
  spherical and rectangular geometries with electrostatic modifier and optional
  drift term.
- Introduce `wall_potential` and `wall_electric_field` mixins plus
  `ChargedWallLossBuilder` with geometry validation and distribution-type
  support; register under `strategy_type="charged"` in the factory.
- Export charged strategy, builder, and charged wall loss rate helper for API
  parity with existing wall loss tooling.

**Affected Components:**
- `particula/dynamics/wall_loss/wall_loss_strategies.py` — charged strategy and
  electrostatic helper logic
- `particula/builder_mixin.py` — wall potential/electric field mixins
- `particula/dynamics/wall_loss/wall_loss_builders.py` — charged builder with
  geometry selection
- `particula/dynamics/wall_loss/wall_loss_factories.py` — factory registration
- `particula/dynamics/wall_loss/rate.py` — charged wall loss rate helper
- `particula/dynamics/wall_loss/__init__.py`, `particula/dynamics/__init__.py`
  — exports
- Tests under `particula/dynamics/wall_loss/tests/` and
  `particula/dynamics/tests/`
- Docs: `docs/Features/wall_loss_strategy_system.md`, example update in
  `docs/Examples/Chamber_Wall_Loss/wall_loss_strategy.md`

### Design Patterns

- Strategy pattern for geometry-aware wall loss with charged extension.
- Builder + factory patterns for ergonomic configuration and validation.
- Vectorized coefficient computation with mask-aware particle-resolved path and
  finite clipping for stability.

### API Surface

- `ChargedWallLossStrategy(wall_eddy_diffusivity, chamber_geometry,
  chamber_radius|chamber_dimensions, wall_potential=0.0,
  wall_electric_field=0.0, distribution_type)`
- `ChargedWallLossBuilder.set_wall_potential(...)` and
  `.set_wall_electric_field(...)` with geometry setters and
  `set_distribution_type(...)`
- Factory: `strategy_type="charged"`
- Rate helper: `get_charged_wall_loss_rate(...)` (parity with strategy output)

## Implementation Tasks

### Backend
- [ ] Implement `ChargedWallLossStrategy` with image-charge, diffusion
      enhancement, optional electric-field drift, and neutral reduction when
      charge/field are zero
- [ ] Add wall potential/electric field mixins and geometry validation to
      `ChargedWallLossBuilder`; register in factory under `charged`
- [ ] Export charged strategy, builder, and rate helper via wall_loss and
      dynamics packages
- [ ] Ensure particle-resolved path clamps survival probabilities and remains
      vectorized

### Tests
- [ ] Charged vs neutral differs for non-zero charge; equals neutral when charge
      is zero (even if wall_potential != 0)
- [ ] Image-charge active when wall_potential == 0 but charge ≠ 0
- [ ] Electric-field drift sign effect on coefficients; zero field matches
      no-drift baseline
- [ ] Geometry validation for spherical vs rectangular and invalid inputs
- [ ] Factory/builder wiring returns charged strategy; distribution type
      propagation verified
- [ ] Particle-resolved survival bounds within [0, 1]
- [ ] Rate helper parity with strategy (if helper added)

### Documentation
- [ ] Docstrings reference electrostatic behavior and cite literature
- [ ] Update wall loss feature docs and example page with charged usage and
      parameter guidance
- [ ] Note neutral reduction path and image-charge behavior when
      `wall_potential == 0`

## Dependencies

- Upstream: #816 (spherical baseline), #817 (rectangular), #818
  (builders/factory), #819 (runnable exports)
- Downstream: Wall loss example consumers and future charged-aware dynamics

## Testing Strategy

### Unit Tests
- Charged vs neutral comparison across geometries and distribution types
- Sign effect (positive vs negative charge) and zero-charge reduction
- Image-charge-only path (wall_potential == 0, charge ≠ 0)
- Electric-field drift on/off with sign-aware behavior
- Geometry validation errors and distribution type validation

### Integration Tests
- Factory returns charged strategy with propagated parameters
- Public exports via `particula.dynamics` succeed
- Rate helper (if present) matches strategy output

### Particle-Resolved Behavior
- Survival probabilities clipped to [0, 1]; concentrations remain
  non-negative after step

## Documentation

- [x] Feature documentation (this file)
- [ ] API documentation updates (charged parameters, rate helper)
- [ ] User guide/example updates (charged wall loss walkthrough)

## Success Criteria

- [ ] `ChargedWallLossStrategy` supports spherical/rectangular geometries and
      all distribution types with neutral reduction when charge/field are zero
- [ ] Image-charge active even when wall_potential == 0; electric-field drift is
      optional and sign-aware
- [ ] Builder + factory expose charged parameters and validate geometry/inputs
- [ ] Particle-resolved survival bounds enforced; outputs finite and
      non-negative
- [ ] Tests cover charged vs neutral, sign effects, zero-potential behavior,
      drift toggling, factory/export wiring
- [ ] Docs/examples updated to guide charged usage
- [ ] All tests passing; code review approved

## Usage Example

```python
import particula as par

strategy = par.dynamics.ChargedWallLossStrategy(
    wall_eddy_diffusivity=1.0e-4,
    chamber_geometry="spherical",
    chamber_radius=0.5,
    wall_potential=0.05,  # V; image-charge applies even when set to 0
    wall_electric_field=0.0,  # V/m (tuple for rectangular)
    distribution_type="particle_resolved",
)

loss_rate = strategy.rate(
    particle=particle,
    temperature=298.0,
    pressure=101325.0,
)

particle = strategy.step(
    particle=particle,
    temperature=298.0,
    pressure=101325.0,
    time_step=1.0,
    sub_steps=2,
)
```

Builder + factory usage:

```python
from particula.dynamics.wall_loss import WallLossFactory

factory = WallLossFactory()
charged = factory.get_strategy(
    strategy_type="charged",
    parameters={
        "wall_eddy_diffusivity": 2.0e-5,
        "chamber_geometry": "rectangular",
        "chamber_dimensions": (1.0, 0.8, 0.6),
        "wall_potential": 0.0,
        "wall_electric_field": (0.0, 0.0, 500.0),
        "distribution_type": "continuous_pdf",
    },
)
```

## Change Log

| Date       | Change                                        | Author       |
|------------|-----------------------------------------------|--------------|
| 2025-12-21 | Initial feature documentation created         | ADW Workflow |

[adr-001]: ../../architecture/decisions/ADR-001-strategy-based-wall-loss-subsystem.md
