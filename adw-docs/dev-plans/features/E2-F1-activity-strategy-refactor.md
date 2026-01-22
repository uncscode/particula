# Feature E2-F1: Activity Strategy Refactor

**Status**: Planning
**Priority**: P1
**Parent Epic**: [E2 - Activity and Equilibria Strategy-Builder-Factory Refactor](../epics/E2-activity-equilibria-refactor.md)
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-01-07
**Size**: Medium (5 phases, ~450 LOC)

## Overview

Refactor the `activity/` module calculations into strategy classes following
the existing `ActivityStrategy` ABC pattern. This feature introduces:

1. **`ActivityNonIdealBinary`**: Strategy wrapping BAT model (Gorkowski 2019)
   for non-ideal organic-water mixture activity calculations

The strategy delegates to existing calculation functions in `activity/`,
maintaining the separation between algorithm implementation and strategy
orchestration.

> **Note**: The Kelvin effect is intentionally NOT included as an activity
> strategy. The Kelvin effect is a surface phenomenon already implemented in
> `SurfaceStrategy.kelvin_term()` and properly applied during condensation
> calculations. Activity strategies compute thermodynamic activities; surface
> strategies handle curvature effects on vapor pressure.

## Acceptance Criteria

- [ ] `ActivityNonIdealBinary` strategy computes BAT model activity
- [ ] `ActivityNonIdealBinaryBuilder` validates required parameters
- [ ] `ActivityFactory` supports `"non_ideal_binary"` type
- [ ] All existing `activity/` tests continue passing
- [ ] New strategies have 80%+ test coverage
- [ ] Docstrings follow Google-style format with examples

## Technical Design

### ActivityNonIdealBinary Strategy

Wraps the BAT model from `activity/activity_coefficients.py`:

```python
class ActivityNonIdealBinary(ActivityStrategy):
    """Non-ideal activity for binary organic-water mixtures (BAT model).
    
    Uses the Binary Activity Thermodynamics model from Gorkowski (2019) to
    compute activity coefficients accounting for non-ideal mixing behavior.
    
    Attributes:
        molar_mass: Organic species molar mass [kg/mol].
        oxygen2carbon: Oxygen to carbon ratio [-].
        density: Species density [kg/m^3].
        functional_group: Optional functional group identifier.
    """
    
    def __init__(
        self,
        molar_mass: float,
        oxygen2carbon: float,
        density: float,
        functional_group: Optional[str] = None,
    ) -> None: ...
    
    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Compute non-ideal activity using BAT model."""
        ...
```

### Builder Classes

```python
class ActivityNonIdealBinaryBuilder(
    BuilderABC, BuilderMolarMassMixin, BuilderDensityMixin
):
    """Builder for ActivityNonIdealBinary strategy."""
    
    def set_oxygen2carbon(self, value: float) -> Self: ...
    def set_functional_group(self, group: Optional[str]) -> Self: ...
    def build(self) -> ActivityNonIdealBinary: ...
```

## Phases

### Phase E2-F1-P0: Activity Module Code Quality Improvements

**Issue**: TBD | **Size**: M | **Status**: Not Started

Improve docstrings, code logic, and function isolation in the existing `activity/`
module to bring it up to particula standards before adding new strategies.

**Tasks**:
- Fix typo: rename `gibbs_free_engery` to `gibbs_free_energy` in `gibbs.py`
- Add type hints to `ratio.py` function docstrings (molar_mass, return types)
- Add `@validate_inputs` decorator to `ratio.py` functions where appropriate
- Refactor `convert_to_oh_equivalent()` to handle additional functional groups
  (currently only supports `None` and `"alcohol"`)
- Add Google-style docstring sections (Args, Returns, Raises, Examples) to:
  - `bat_blending.py`: `bat_blending_weights()`, `_calculate_blending_weights()`
  - `bat_coefficients.py`: `coefficients_c()`
  - `gibbs.py`: `gibbs_free_energy()` (after rename)
- Ensure all public functions have usage examples in docstrings
- Add unit tests for any edge cases discovered during refactoring
- Run linters (ruff, mypy) and fix any issues

**Files Modified**:
- `particula/activity/gibbs.py`
- `particula/activity/ratio.py`
- `particula/activity/convert_functional_group.py`
- `particula/activity/bat_blending.py`
- `particula/activity/bat_coefficients.py`
- `particula/activity/tests/gibbs_test.py` (update for rename)
- `particula/activity/tests/ratio_test.py` (add edge case tests)

---

### Phase E2-F1-P1: Create ActivityNonIdealBinary Strategy

**Issue**: TBD | **Size**: M | **Status**: Not Started

Create the `ActivityNonIdealBinary` strategy class that wraps BAT model
functions for non-ideal organic-water activity calculations.

**Tasks**:
- Create `ActivityNonIdealBinary` class in `particles/activity_strategies.py`
- Implement `activity()` method calling `bat_activity_coefficients()`
- Implement `get_name()` returning strategy identifier
- Add comprehensive docstrings with examples and references
- Add unit tests in `particles/tests/activity_strategies_test.py`

**Files Modified**:
- `particula/particles/activity_strategies.py`
- `particula/particles/tests/activity_strategies_test.py`

---

### Phase E2-F1-P2: Create ActivityNonIdealBinaryBuilder

**Issue**: TBD | **Size**: M | **Status**: Not Started

Create the builder class for `ActivityNonIdealBinary` with parameter
validation and fluent interface.

**Tasks**:
- Create `ActivityNonIdealBinaryBuilder` in `particles/activity_builders.py`
- Inherit from `BuilderABC`, `BuilderMolarMassMixin`, `BuilderDensityMixin`
- Add `set_oxygen2carbon()` method with validation
- Add `set_functional_group()` method
- Implement `build()` with pre-build validation
- Add unit tests in `particles/tests/activity_builders_test.py`

**Files Modified**:
- `particula/particles/activity_builders.py`
- `particula/particles/tests/activity_builders_test.py`

---

### Phase E2-F1-P3: Update ActivityFactory

**Issue**: TBD | **Size**: S | **Status**: Not Started

Update `ActivityFactory` to support new strategy types via string keys.

**Tasks**:
- Add `"non_ideal_binary"` mapping to `ActivityNonIdealBinaryBuilder`
- Update factory docstrings with new strategy types
- Add factory dispatch tests for new strategies
- Verify backward compatibility with existing strategy types

**Files Modified**:
- `particula/particles/activity_factories.py`
- `particula/particles/tests/activity_factories_test.py`

---

### Phase E2-F1-P4: Activity Module Function Isolation and Cleanup

**Issue**: TBD | **Size**: M | **Status**: Not Started

Improve function isolation and code organization in `activity/` module for
maintainability and testability.

**Tasks**:
- Extract helper functions from `gibbs_mixing.py`:
  - Separate weighted calculation logic into smaller functions
  - Reduce cognitive complexity of `gibbs_mix_weight()` (currently ~50 lines)
- Add missing docstring parameters to `activity_coefficients.py`:
  - Document return tuple elements more explicitly
  - Add mathematical equations in docstring references
- Improve `phase_separation.py` code organization:
  - Add explicit return type annotations to all functions
  - Document the physical meaning of `MIN_SPREAD_IN_AW` and `Q_ALPHA_AT_1PHASE_AW`
- Validate all `__init__.py` exports are current and complete
- Add missing `__all__` exports to `activity/__init__.py`
- Run full test suite and verify 80%+ coverage maintained
- Update any tests that need adjustment after refactoring

**Files Modified**:
- `particula/activity/__init__.py`
- `particula/activity/gibbs_mixing.py`
- `particula/activity/activity_coefficients.py`
- `particula/activity/phase_separation.py`
- `particula/activity/tests/` (various test updates)

## Dependencies

- **Internal**: `ActivityStrategy` ABC, `BuilderABC`, mixins
- **External**: None
- **Blocks**: E2-F3 (Integration and Documentation)

## Testing Strategy

### Unit Tests
- Test each strategy's `activity()` method with known inputs
- Test builder validation (required params, value constraints)
- Test factory dispatch for all strategy types

### Integration Tests
- Verify strategies work with `ParticleRepresentation`
- Verify strategies produce consistent results with direct function calls

### Regression Tests
- Ensure existing `activity/` module tests pass unchanged
- Verify existing activity strategies still function correctly

## References

- Gorkowski, K., Preston, T. C., & Zuend, A. (2019). Relative-humidity-dependent
  organic aerosol thermodynamics via an efficient reduced-complexity model.
  Atmospheric Chemistry and Physics. https://doi.org/10.5194/acp-19-13383-2019

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-07 | Initial feature creation | ADW |
| 2026-01-07 | Added P0 code quality phase and P6 function isolation phase | ADW |
| 2026-01-08 | Removed ActivityKelvinEffect phases (P3, P4) - Kelvin effect already implemented in SurfaceStrategy | ADW |

