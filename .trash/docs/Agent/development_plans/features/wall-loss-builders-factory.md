# Feature: Wall Loss Builders, Mixins, and Factory

**Status:** In Progress
**Priority:** P1
**Assignees:** ADW Workflow
**Labels:** feature, dynamics, wall-loss, builders
**Milestone:** v0.2.x
**Size:** M (~180 LOC core + tests)

**Start Date:** 2025-12-18
**Target Date:** 2026-01-08
**Created:** 2025-12-18
**Updated:** 2025-12-18

**Related Issues:** [#818](https://github.com/uncscode/particula/issues/818) (parent: [#72](https://github.com/uncscode/particula/issues/72))
**Related PRs:** (pending)
**Related ADRs:** [ADR-001][adr-001]

---

## Overview

Add builder-pattern support, reusable validation/unit mixins, and a factory for
wall loss strategies so wall-loss geometry selection aligns with other dynamics
modules. Builders provide unit-aware setters for eddy diffusivity, chamber
geometry, and distribution type, and feed into a factory that returns
strategies by name while enforcing validation.

### Problem Statement

Wall loss strategies lacked builder/factory parity with other dynamics systems,
forcing direct strategy instantiation and manual unit handling. Without shared
validation and unit conversion, users risk inconsistent inputs (e.g., cm vs m or
invalid distribution types) and duplicated glue code.

### Value Proposition

- Aligns wall loss with existing builder/factory ergonomics across dynamics.
- Centralizes validation (positive inputs, rectangular dimensions length == 3)
and unit conversion for safer usage.
- Enables users to choose geometries via a single `WallLossFactory` entry point
with consistent distribution-type handling.

## Phases

> **Philosophy:** Each phase should be one GitHub issue with a focused scope
> (~100 lines of code or less, excluding tests/docs).

- [ ] **Phase 1:** Add wall loss builders, mixins, factory, exports, and tests
  - GitHub Issue: #818
  - Status: In Progress
  - Size: M (~180 LOC core + tests)
  - Dependency: Parent epic #72 (wall loss)
  - Estimated Effort: 3–4 days

## User Stories

### Story 1: Builder parity for wall loss
**As a** dynamics contributor
**I want** builder/factory coverage for wall loss geometries
**So that** users configure wall loss like other dynamics modules without
special-case code

**Acceptance Criteria:**
- [ ] Builders expose unit-aware setters for diffusivity and geometry
- [ ] Builders validate required parameters before build
- [ ] Factory returns spherical and rectangular strategies by name
- [ ] Errors surface when strategy type is unknown

### Story 2: Safe rectangular geometry configuration
**As a** simulation user
**I want** validation of chamber dimensions and distribution type
**So that** I avoid silent misconfiguration and unit mistakes

**Acceptance Criteria:**
- [ ] `chamber_dimensions` must be length 3 with all positive values
- [ ] `distribution_type` accepts only discrete, continuous_pdf,
      particle_resolved
- [ ] Non-positive eddy diffusivity or radius/dimensions raise ValueError
- [ ] Unit conversion supports cm-to-m and similar length/diffusivity units

## Technical Approach

### Architecture Changes

- Add mixins in `particula/builder_mixin.py` for wall eddy diffusivity, chamber
  radius, chamber dimensions, and distribution type with validation +
  `get_unit_conversion`.
- Create `particula/dynamics/wall_loss/wall_loss_builders.py` implementing
  `SphericalWallLossBuilder` and `RectangularWallLossBuilder` using BuilderABC.
- Create `particula/dynamics/wall_loss/wall_loss_factories.py` implementing
  `WallLossFactory` via `StrategyFactoryABC` mapping `spherical` and
  `rectangular` builders.
- Export builders and factory from `particula.dynamics.wall_loss` and
  `particula.dynamics` for parity with other dynamics APIs.

**Affected Components:**
- `particula/builder_mixin.py` — new wall-loss mixins with unit conversion and
  validation
- `particula/dynamics/wall_loss/wall_loss_builders.py` — geometry-specific
  builders
- `particula/dynamics/wall_loss/wall_loss_factories.py` — factory mapping to
  builders
- `particula/dynamics/wall_loss/__init__.py` and `particula/dynamics/__init__.py`
  — exports
- `particula/dynamics/wall_loss/tests/`, `particula/dynamics/tests/` — new test
  coverage

### Design Patterns

- Builder pattern with fluent, unit-aware setters and `pre_build_check`.
- StrategyFactoryABC usage for consistent `get_strategy` behavior and error
  handling.
- Shared validation mixins to avoid duplicated geometry/distribution checks.

### Data Model Changes

No persistent data model changes; builders/factory are runtime-only helpers.

### API Changes

New public APIs:
- `particula.dynamics.wall_loss.SphericalWallLossBuilder`
- `particula.dynamics.wall_loss.RectangularWallLossBuilder`
- `particula.dynamics.wall_loss.WallLossFactory`
- Re-exports under `particula.dynamics.*` for parity

Distribution type options (validated): `"discrete"`, `"continuous_pdf"`,
`"particle_resolved"`.

Expected units:
- `wall_eddy_diffusivity`: base m^2/s (supports e.g., cm^2/s via conversion)
- `chamber_radius`: meters (supports cm, mm via conversion)
- `chamber_dimensions`: tuple of three lengths in meters after conversion

### Usage Examples

Builder chaining with unit conversion:

```python
import particula as par

builder = par.dynamics.SphericalWallLossBuilder()
strategy = (
    builder
    .set_wall_eddy_diffusivity(1.2e-3, units="cm^2/s")
    .set_chamber_radius(50.0, units="cm")
    .set_distribution_type("discrete")
    .build()
)
```

Factory creation by name with parameters dict:

```python
from particula.dynamics.wall_loss import WallLossFactory

factory = WallLossFactory()
rectangular = factory.get_strategy(
    strategy_type="rectangular",
    parameters={
        "wall_eddy_diffusivity": 2.5e-5,
        "wall_eddy_diffusivity_units": "m^2/s",
        "chamber_dimensions": (200.0, 150.0, 100.0),
        "chamber_dimensions_units": "cm",
        "distribution_type": "continuous_pdf",
    },
)
```

## Implementation Tasks

> For single-phase features, list all tasks here. For multi-phase features,
> provide high-level tasks and link to phase-specific files for detailed tasks.

### Backend Tasks
- [ ] Add four wall-loss mixins with unit conversion and validation to
      `builder_mixin.py`
- [ ] Implement `wall_loss_builders.py` with spherical/rectangular builders and
      required parameter enforcement
- [ ] Implement `wall_loss_factories.py` using StrategyFactoryABC mappings
- [ ] Wire exports in `wall_loss/__init__.py` and `dynamics/__init__.py`
- [ ] Add tests for builders, factory, and re-exports covering unit conversion,
      validation, and distribution type handling

**Estimated Effort:** 3–4 days

### Frontend Tasks
- [ ] None (backend-only feature)

**Estimated Effort:** N/A

### Database Tasks
- [ ] None (no persistence changes)

**Estimated Effort:** N/A

### Infrastructure Tasks
- [ ] None (no infra changes required)

**Estimated Effort:** N/A

## Dependencies

### Upstream Dependencies
- BuilderABC and StrategyFactoryABC patterns for builder/factory behavior
- `get_unit_conversion` utilities for unit handling

### Downstream Dependencies
- Wall loss strategy users in dynamics needing builder/factory ergonomics
- Future geometries leveraging shared wall-loss mixins

### External Dependencies
- None

## Blockers

- [ ] None identified

## Testing Strategy

### Unit Tests
- Builder setters validate positivity and convert units (cm->m) for diffusivity
  and geometry
- `chamber_dimensions` validation ensures length == 3 and all positive
- `distribution_type` rejects invalid values with clear error message
- Builders support method chaining and `pre_build_check` enforcement

**Test Cases:**
- [ ] Spherical builder build success and missing-parameter failure
- [ ] Rectangular builder unit conversion and dimension-length failure
- [ ] Distribution type validation across allowed/invalid inputs
- [ ] Negative/zero geometry or diffusivity raises ValueError

### Integration Tests
- Factory returns spherical and rectangular strategies via `get_strategy`
- Unknown strategy type raises ValueError (StrategyFactoryABC behavior)
- Re-export smoke tests via `particula.dynamics`

**Test Cases:**
- [ ] Factory parameters propagate to built strategies
- [ ] Unknown strategy type error messaging
- [ ] Dynamics-level imports expose builders and factory

### End-to-End Tests
Not required; feature is internal factory/builder wiring.

### Performance Tests
Not required; builder/factory overhead is minimal.

## Documentation

- [x] Feature documentation (this file)
- [ ] API documentation updates (if builder/factory reference pages are added)
- [ ] User guide updates (examples/notebooks)
- [ ] Create migration guide (not needed unless API changes further)
- [ ] Update additional guides if factory usage is added elsewhere

## Security Considerations

- Input validation guards against invalid numerical values; no external I/O.
- No new data persistence or credentials; standard ValueError paths only.

**Security Checklist:**
- [ ] Validate external inputs (n/a; internal builder usage)
- [ ] Avoid unsafe eval/exec (n/a)
- [ ] Ensure error messages do not leak sensitive data (n/a)

## Performance Considerations

- Builder/factory overhead is negligible; ensure unit conversion helpers are
  reused and not recomputed unnecessarily.
- Strategy creation should remain lightweight for per-simulation configuration.

**Performance Targets:**
- No measurable overhead compared to direct strategy instantiation
- Unit conversion performed once per setter invocation

## Rollout Strategy

### Deployment Plan
Ship with next minor release after tests pass; no migration required.

### Feature Flags
- None (fully enabled once merged)

### Rollback Plan
- Revert builder/factory modules and exports; existing direct strategy paths
  remain intact.

## Success Criteria

- [ ] Builders expose validated, unit-aware setters for diffusivity and geometry
- [ ] `WallLossFactory` returns spherical/rectangular strategies and errors on
      unknown types
- [ ] Distribution type limited to discrete, continuous_pdf, particle_resolved
- [ ] Tests cover validation, unit conversion, factory mapping, and exports
- [ ] All tests passing
- [ ] Code review approved
- [ ] Documentation updated
- [ ] Performance targets met
- [ ] Security review passed

## Metrics to Track

- Unit test coverage for new modules ≥ 95%
- Factory/Builder error branch coverage present
- No regression in wall loss strategy import paths

## Timeline

| Phase/Milestone | Start Date | Target Date | Actual Date | Status |
|----------------|------------|-------------|-------------|--------|
| Phase 1 - Builders, mixins, factory, exports, tests | 2025-12-18 | 2026-01-08 | — | In Progress |

## Open Questions

- Should we expose helper defaults for common diffusivity units in docs/examples?
- Do we need a helper for unit validation error messaging consistency across
  builders?
- Should future geometries (e.g., cylindrical) reuse the same mixins?

## Notes

- Distribution types allowed: `discrete`, `continuous_pdf`, `particle_resolved`.
- Rectangular dimensions must be a length-3 tuple/list with all positive values.
- Unit conversion is handled per setter via `get_unit_conversion`; pass
  `*_units` in parameter dicts when using the factory.

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-18 | Initial feature documentation created | ADW Workflow |

[adr-001]: ../../architecture/decisions/ADR-001-strategy-based-wall-loss-subsystem.md
