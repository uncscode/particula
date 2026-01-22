# Feature E2-F3: Integration and Documentation

**Status**: Planning
**Priority**: P1
**Parent Epic**: [E2 - Activity and Equilibria Strategy-Builder-Factory Refactor](../epics/E2-activity-equilibria-refactor.md)
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-01-07
**Size**: Medium (4 phases, ~300 LOC + documentation)

## Overview

Integrate the new activity strategies and equilibria runnable into the main
particula module exports and create comprehensive documentation including:

1. **Module Integration**: Export new classes through `particles/` and
   `equilibria/` package `__init__.py` files
2. **Theory Documentation**: Explain activity models, Kelvin effect, and
   equilibria calculations in `docs/Theory/`
3. **Feature Documentation**: User-facing documentation in `docs/Features/`
4. **Examples**: Practical usage examples in `docs/Examples/`

## Acceptance Criteria

- [ ] New activity strategies accessible via `particula.particles` namespace
- [ ] Equilibria runnable accessible via `particula.equilibria` namespace
- [ ] Theory documentation covers BAT model, Kelvin effect, and partitioning
- [ ] Feature documentation explains activity system with examples
- [ ] Example notebooks/scripts demonstrate common use cases
- [ ] All documentation links validate correctly
- [ ] Development plan indexes updated

## Phases

### Phase E2-F3-P1: Module Integration and Exports

**Issue**: TBD | **Size**: S | **Status**: Not Started

Update module `__init__.py` files to export new classes through the main
particula namespace.

**Tasks**:
- Update `particula/particles/__init__.py` with new activity exports:
  - `ActivityNonIdealBinary`
  - `ActivityNonIdealBinaryBuilder`
  - `ActivityKelvinEffect`
  - `ActivityKelvinEffectBuilder`
- Update `particula/equilibria/__init__.py` with new exports:
  - `EquilibriaStrategy`
  - `LiquidVaporPartitioningStrategy`
  - `LiquidVaporPartitioningBuilder`
  - `EquilibriaFactory`
  - `Equilibria`
- Update main `particula/__init__.py` if needed
- Verify import paths work correctly
- Add import tests to verify exports

**Files Modified**:
- `particula/particles/__init__.py`
- `particula/equilibria/__init__.py`
- `particula/__init__.py` (if needed)
- `particula/tests/import_test.py` (new or update)

---

### Phase E2-F3-P2: Theory Documentation

**Issue**: TBD | **Size**: M | **Status**: Not Started

Create theory documentation explaining the scientific basis for activity
calculations and equilibria solving.

**Tasks**:
- Create `docs/Theory/Activity_Calculations/` directory
- Create `docs/Theory/Activity_Calculations/index.md` section overview
- Create `activity_theory.md` explaining:
  - Raoult's Law and ideal activity
  - Non-ideal activity and activity coefficients
  - BAT model (Binary Activity Thermodynamics)
  - Kelvin effect and curvature correction
- Create `equilibria_theory.md` explaining:
  - Liquid-vapor partitioning
  - Phase separation (alpha/beta phases)
  - Volatility basis set concepts
- Add Mermaid diagrams for:
  - Activity calculation flow
  - Equilibria solving process
- Update `docs/Theory/index.md` with new sections

**Files Modified**:
- `docs/Theory/Activity_Calculations/index.md` (new)
- `docs/Theory/Activity_Calculations/activity_theory.md` (new)
- `docs/Theory/Activity_Calculations/equilibria_theory.md` (new)
- `docs/Theory/index.md`

---

### Phase E2-F3-P3: Feature Documentation and Examples

**Issue**: TBD | **Size**: M | **Status**: Not Started

Create feature documentation and practical examples for activity and
equilibria calculations.

**Tasks**:
- Create `docs/Features/activity_system.md` documenting:
  - Available activity strategies
  - Builder usage patterns
  - Factory usage for dynamic strategy selection
  - Integration with particle representations
- Create `docs/Examples/Activity/` directory with examples:
  - `ideal_activity_example.py` - Basic ideal activity usage
  - `bat_activity_example.py` - Non-ideal BAT model example
  - `kelvin_effect_example.py` - Kelvin correction example
  - `equilibria_example.py` - Liquid-vapor partitioning example
- Create Jupyter notebook `activity_tutorial.ipynb` with:
  - Interactive walkthrough of activity concepts
  - Visualization of activity vs composition
  - Comparison of different activity models
- Update `docs/Features/index.md` and `docs/Examples/index.md`

**Files Modified**:
- `docs/Features/activity_system.md` (new)
- `docs/Examples/Activity/ideal_activity_example.py` (new)
- `docs/Examples/Activity/bat_activity_example.py` (new)
- `docs/Examples/Activity/kelvin_effect_example.py` (new)
- `docs/Examples/Activity/equilibria_example.py` (new)
- `docs/Examples/Activity/activity_tutorial.ipynb` (new)
- `docs/Features/index.md`
- `docs/Examples/index.md`

---

### Phase E2-F3-P4: Development Documentation Update

**Issue**: TBD | **Size**: XS | **Status**: Not Started

Update development plan indexes and mark epic as complete.

**Tasks**:
- Update `adw-docs/dev-plans/epics/index.md`:
  - Move E2 to Active Epics section (or Completed when done)
  - Update next available ID
- Update `adw-docs/dev-plans/features/index.md`:
  - Add E2-F1, E2-F2, E2-F3 to appropriate sections
  - Update status as phases complete
- Update `adw-docs/dev-plans/README.md` if needed
- Add completion notes and lessons learned to epic document
- Validate all markdown links in documentation

**Files Modified**:
- `adw-docs/dev-plans/epics/index.md`
- `adw-docs/dev-plans/features/index.md`
- `adw-docs/dev-plans/epics/E2-activity-equilibria-refactor.md`
- `adw-docs/dev-plans/README.md` (if needed)

## Dependencies

- **Depends On**: E2-F1, E2-F2 (must be complete before P1)
- **Internal**: None
- **External**: MkDocs for documentation rendering

## Testing Strategy

### Documentation Tests
- Validate all internal markdown links
- Verify code examples compile/run
- Test Jupyter notebook execution

### Integration Tests
- Verify all exports are accessible
- Test import statements from user perspective:
  ```python
  import particula as par
  strategy = par.particles.ActivityNonIdealBinary(...)
  equilibria = par.equilibria.Equilibria(...)
  ```

## Documentation Structure

```
docs/
├── Theory/
│   ├── Activity_Calculations/
│   │   ├── activity_theory.md      # Theory: activity models
│   │   └── equilibria_theory.md    # Theory: partitioning
│   └── index.md                    # Updated
├── Features/
│   ├── activity_system.md          # Feature: activity strategies
│   └── index.md                    # Updated
└── Examples/
    ├── Activity/
    │   ├── ideal_activity_example.py
    │   ├── bat_activity_example.py
    │   ├── kelvin_effect_example.py
    │   ├── equilibria_example.py
    │   └── activity_tutorial.ipynb
    └── index.md                    # Updated
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-07 | Initial feature creation | ADW |
| 2026-01-07 | Added Activity_Calculations/index.md to P2 tasks | ADW |

