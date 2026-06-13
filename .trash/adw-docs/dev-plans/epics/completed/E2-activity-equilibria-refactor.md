# Epic E2: Activity and Equilibria Strategy-Builder-Factory Refactor

**Status**: Completed
**Priority**: P1
**Owners**: TBD
**Start Date**: 2026-01-07
**Completion Date**: 2026-01-21
**Last Updated**: 2026-01-21
**Size**: Large (3 features, ~15 phases)

## Vision

Refactor the `activity/` and `equilibria/` modules to align with particula's
Strategy-Builder-Factory design pattern, enabling consistent, extensible, and
testable activity calculations. This epic introduces:

1. **New Activity Strategy**: `ActivityNonIdealBinary` (BAT model) following
   the existing `ActivityStrategy` ABC
2. **Equilibria as Runnable**: Transform liquid-vapor partitioning into a
   `Runnable` pattern similar to `dynamics/` for solving equilibrium states
3. **Documentation**: Comprehensive examples, theory, and feature documentation

The refactor preserves existing calculation functions while wrapping them in
strategy classes that integrate with the builder and factory system.

> **Note**: The Kelvin effect is NOT part of this refactor. It is a surface
> phenomenon already implemented in `SurfaceStrategy.kelvin_term()` and properly
> applied during condensation calculations in `dynamics/condensation/`.

## Scope

### In Scope

- **Activity Module Refactor**:
  - Create `ActivityNonIdealBinary` strategy wrapping BAT model functions
  - Create corresponding builder with validation and mixins
  - Add strategy entry to `ActivityFactory`
  - Keep core calculation functions in `activity/` module (strategies call them)

- **Equilibria Module Refactor**:
  - Create `EquilibriaRunnable` following `Runnable` ABC pattern
  - Create `LiquidVaporPartitioningStrategy` for equilibrium solving
  - Create builders and factory for equilibria strategies
  - Integrate with activity strategies for non-ideal thermodynamics

- **Integration**:
  - Update `particles/activity_strategies.py` with new strategies
  - Update `particles/activity_builders.py` with new builders
  - Update `particles/activity_factories.py` with new factory entries
  - Ensure backward compatibility with existing activity usage

- **Documentation**:
  - Add theory documentation for activity calculations (`docs/Theory/`)
  - Add feature documentation (`docs/Features/`)
  - Add practical examples (`docs/Examples/`)
  - Update docstrings throughout refactored modules

### Out of Scope

- Changes to coagulation or condensation dynamics (separate concerns)
- New thermodynamic models beyond BAT (future work)
- Performance optimization (can be addressed in maintenance)
- Breaking changes to existing public APIs

## Dependencies

- **Internal**: Existing `ActivityStrategy` ABC, `BuilderABC`, `StrategyFactoryABC`
- **External**: None
- **Blockers**: None identified

## Features

| ID | Name | Priority | Phases | Status |
|----|------|----------|--------|--------|
| E2-F1 | [Activity Strategy Refactor](../features/completed/E2-F1-activity-strategy-refactor.md) | P1 | 5 | Completed |
| E2-F2 | [Equilibria Runnable Refactor](../features/completed/E2-F2-equilibria-runnable-refactor.md) | P2 | 6 | Completed |
| E2-F3 | [Integration and Documentation](../features/completed/E2-F3-integration-documentation.md) | P1 | 4 | Completed |

## Phase Overview

### E2-F1: Activity Strategy Refactor (5 phases)
- P0: **Code Quality** - Improve docstrings, fix typos, add validation to `activity/` module
- P1: Create `ActivityNonIdealBinary` strategy wrapping BAT functions
- P2: Create `ActivityNonIdealBinaryBuilder` with validation
- P3: Update `ActivityFactory` with new strategy entry
- P4: **Function Isolation** - Refactor complex functions, improve organization

### E2-F2: Equilibria Runnable Refactor (6 phases)
- P0: **Code Quality** - Refactor `partitioning.py` (extract helpers, add docstrings)
- P1: Create `EquilibriaStrategy` ABC and `LiquidVaporPartitioningStrategy`
- P2: Create `LiquidVaporPartitioningBuilder` with validation
- P3: Create `EquilibriaFactory` for strategy selection
- P4: Create `Equilibria` runnable following `Runnable` pattern
- P5: **Module Cleanup** - Final exports and backward compatibility

### E2-F3: Integration and Documentation (4 phases)
- P1: Integrate new strategies into `particles/` module exports
- P2: Add theory documentation for activity calculations
- P3: Add feature documentation and examples
- P4: Update development documentation and indexes

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds as shipped (80%+)
- **Self-Contained Tests**: Each phase includes `*_test.py` files
- **Test-First Completion**: Tests pass before phase completion
- **Backward Compatibility**: Existing tests must continue passing

## Testing Strategy

### Unit Tests
- `particula/particles/tests/activity_strategies_test.py` - Strategy behavior
- `particula/particles/tests/activity_builders_test.py` - Builder validation
- `particula/particles/tests/activity_factories_test.py` - Factory dispatch
- `particula/equilibria/tests/` - Equilibria strategy and runnable tests

### Integration Tests
- Verify new strategies integrate with existing particle representations
- Verify equilibria runnable works with different activity strategies
- Verify backward compatibility with existing usage patterns

### Existing Test Suites (must pass)
- `particula/activity/tests/` - All 8 existing test files
- `particula/equilibria/tests/partitioning_test.py` - Existing partitioning test

## Architecture Notes

### Strategy Pattern
Strategies encapsulate activity calculation algorithms behind the `ActivityStrategy`
interface. The `activity()` method returns species activities, and
`partial_pressure()` computes surface partial pressures.

```
ActivityStrategy (ABC)
├── ActivityIdealMolar (existing)
├── ActivityIdealMass (existing)
├── ActivityIdealVolume (existing)
├── ActivityKappaParameter (existing)
└── ActivityNonIdealBinary (new - BAT model)
```

### Builder Pattern
Builders provide fluent interfaces for constructing strategies with validation.
Each builder inherits from `BuilderABC` and uses mixins for common parameters.

### Factory Pattern
Factories map string keys to builders, enabling dynamic strategy selection:
```python
factory = ActivityFactory()
strategy = factory.get_strategy("non_ideal_binary", parameters)
```

### Runnable Pattern (Equilibria)
Equilibria follows the `Runnable` pattern from `dynamics/`:
```python
equilibria = EquilibriaRunnable(strategy=LiquidVaporPartitioningStrategy(...))
result = equilibria.execute(aerosol, temperature, pressure)
```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing tests | Low | High | Run full test suite each phase |
| API inconsistency | Medium | Medium | Follow existing naming conventions |
| Documentation gaps | Medium | Low | Dedicated documentation phase |

## Completion Notes

### Summary
- Delivered ActivityNonIdealBinary strategy and builder plus factory integration for activity module.
- Delivered equilibria runnable, strategies, and builders following Strategy-Builder-Factory-Runnable pattern.
- Added integration and documentation assets across theory, features, and examples to close the epic.

### Deviations from Plan
- Kelvin effect remained scoped to SurfaceStrategy; no separate activity strategy added.
- Documentation volume exceeded initial estimate but remained within XS change budget.

### Lessons Learned
- Aligning strategy/builder/factory patterns across modules simplifies exports and testing.
- Early link validation prevents regressions when moving dev-plan assets to completed.

### Actual vs Planned
- Shipped all three features with planned phases; no scope deferrals.
- Maintained backward-compatible docs and links after relocation to completed/.

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-07 | Initial epic creation | ADW |
| 2026-01-07 | Added code quality phases (P0) and cleanup phases to E2-F1 and E2-F2 | ADW |
| 2026-01-08 | Removed ActivityKelvinEffect from E2-F1 (already in SurfaceStrategy); 7→5 phases | ADW |
| 2026-01-21 | Marked epic completed; added completion notes and updated feature links/statuses | ADW |

