# Feature: Charge Conservation in Coagulation Pairs

**Status:** Done
**Priority:** P2 (Medium)
**Assignees:** ADW Workflow
**Labels:** feature, dynamics, coagulation, particle-resolved
**Milestone:** v0.2.7
**Size:** M (~100 LOC)

**Start Date:** 2025-12-02
**Target Date:** 2025-12-02
**Created:** 2025-12-02
**Updated:** 2025-12-02

**Related Issues:** #823
**Related PRs:** (pending)
**Related ADRs:** None

---

## Overview

This feature adds charge conservation during particle-resolved coagulation events. When particles collide in particle-resolved simulations, their masses are combined but previously charges were not handled. This implementation adds charge addition similar to mass addition, enabling physically accurate charge conservation in particle-resolved coagulation simulations.

### Problem Statement

In particle-resolved coagulation simulations, particles carry charge information that affects their collision behavior. When two particles collide and merge, the resulting particle should conserve the total charge of the colliding pair. Without this feature, charge information is lost or becomes inconsistent after coagulation events, leading to physically inaccurate simulation results.

### Value Proposition

- **Physical Accuracy:** Enables accurate simulation of charged particle dynamics during coagulation
- **Electrostatic Modeling:** Supports research on electrostatic effects in aerosol systems
- **Backward Compatible:** Works seamlessly with existing code that doesn't use charges

## Phases

> **Philosophy:** Each phase should be one GitHub issue with a focused scope (~100 lines of code or less, excluding tests/docs). Small, focused changes make reviews smooth and safe. Smooth is safe, and safe is fast.

- [x] **Phase 1:** Charge Conservation Implementation - Add charge handling to collide_pairs methods
  - GitHub Issue: #823
  - Status: Complete
  - Size: M (~100 lines of code, ~40 excluding tests)
  - Dependency: None
  - Estimated Effort: 1 day

## User Stories

### Story 1: Charged Particle Coagulation Simulation
**As a** aerosol researcher
**I want** particle charges to be conserved when particles collide during coagulation
**So that** I can accurately model electrostatic effects in particle-resolved simulations

**Acceptance Criteria:**
- [x] Charges are summed when two particles collide (algebraic sum preserves sign)
- [x] Zero-charge particles handle collisions correctly (charge transferred to merged particle)
- [x] Simulations without charge arrays continue to work (backward compatibility)

### Story 2: Performance-Optimized Charge Handling
**As a** simulation developer
**I want** charge handling to be efficient and not impact performance when charges are zero or absent
**So that** simulations without charge effects don't incur unnecessary computational overhead

**Acceptance Criteria:**
- [x] No charge operations when charge array is None
- [x] Skip charge processing when all colliding pairs have zero charge
- [x] Performance impact minimal for uncharged particle systems

## Technical Approach

### Architecture Changes

The implementation follows **Option A** from the issue discussion: adding charge handling directly to the strategy method, keeping all collision logic together and maintaining clean separation of concerns.

**Affected Components:**
- `DistributionStrategy` (base.py) - Abstract method signature updated to optionally accept and return charge array
- `ParticleResolvedSpeciatedMass` - Main charge handling implementation in `collide_pairs()`
- `ParticleRepresentation` - Pass/receive charge array through strategy
- Moving bin strategies - Signature updates for interface compliance

### Design Patterns

- **Strategy Pattern:** Charge handling follows the existing strategy pattern for distribution operations
- **Optional Parameter Pattern:** Charge array is optional to maintain backward compatibility
- **Early Return Optimization:** Skip charge processing when not needed

### API Changes

**Modified Method Signatures:**

```python
# DistributionStrategy.collide_pairs() - base.py
def collide_pairs(
    self,
    distribution: NDArray,
    concentration: NDArray,
    density: NDArray,
    indices: NDArray,
    charge: Optional[NDArray] = None,  # NEW
) -> tuple[NDArray, NDArray, Optional[NDArray]]:  # NEW return type
```

## Implementation Tasks

### Core Implementation Tasks
- [x] Update `DistributionStrategy.collide_pairs()` abstract method signature in `base.py`
- [x] Update `ParticleResolvedSpeciatedMass.collide_pairs()` to handle charge addition
- [x] Update `ParticleRepresentation.collide_pairs()` to pass charge array and update `self.charge`
- [x] Update `MassBasedMovingBin.collide_pairs()` signature
- [x] Update `RadiiBasedMovingBin.collide_pairs()` signature
- [x] Update `SpeciatedMassMovingBin.collide_pairs()` signature

### Performance Optimization Tasks
- [x] Add early return if charge array is None
- [x] Check only colliding pairs' charges (not entire array) for efficiency

**Estimated Effort:** 1 day

## Dependencies

### Upstream Dependencies
- None - All changes are internal to the `particula` package

### Downstream Dependencies
- Coagulation strategy classes that call `particle.collide_pairs()` - No changes needed (signature unchanged at representation level)

## Testing Strategy

### Unit Tests
Test charge conservation at the strategy level with various scenarios.

**Test Cases:**
- [x] `test_collide_pairs_with_charge()` - Mixed positive/negative charges sum correctly
- [x] `test_collide_pairs_charge_one_zero()` - One particle has zero charge
- [x] `test_collide_pairs_no_charge()` - Backward compatibility with None charge
- [x] `test_collide_pairs_zero_charge_optimization()` - All-zero charges are no-op
- [x] `test_collide_pairs_1d_distribution_with_charge()` - 1D distribution support
- [x] `test_collide_pairs_representation_with_charge()` - Integration through ParticleRepresentation

### Integration Tests
- [x] Charge conservation through full `ParticleRepresentation.collide_pairs()` flow
- [x] Verify charge array shape is preserved after collisions

## Documentation

- [x] Feature documentation (this file)
- [x] Update docstrings for modified methods
- [x] Add inline comments explaining charge handling logic

## Performance Considerations

The implementation includes two key optimizations:

1. **None Check:** If charge array is None, no charge processing occurs
2. **Colliding Pairs Check:** Only check charges in colliding pairs (not entire array) before processing

**Performance Targets:**
- Zero overhead for simulations without charges
- Minimal overhead for simulations with all-zero charges

## Success Criteria

- [x] Charge conservation implemented in `collide_pairs()` methods
- [x] Performance optimization: only process when charge array present AND non-zero in colliding pairs
- [x] Backward compatible: works with existing code (None charge array)
- [x] All tests passing (6 new tests added)
- [x] Code review approved
- [x] Documentation updated
- [x] Linting passes

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `particula/particles/distribution_strategies/base.py` | Abstract method signature update | ~10 LOC |
| `particula/particles/distribution_strategies/particle_resolved_speciated_mass.py` | Main charge handling implementation | ~25 LOC |
| `particula/particles/distribution_strategies/mass_based_moving_bin.py` | Signature update | ~5 LOC |
| `particula/particles/distribution_strategies/radii_based_moving_bin.py` | Signature update | ~5 LOC |
| `particula/particles/distribution_strategies/speciated_mass_moving_bin.py` | Signature update | ~5 LOC |
| `particula/particles/representation.py` | Pass/receive charge array | ~5 LOC |
| `particula/particles/distribution_strategies/tests/particle_resolved_speciated_mass_test.py` | 6 new tests | ~60 LOC |

**Total:** ~115 lines (excluding tests: ~55 lines)

## Edge Cases Handled

1. **Empty indices array:** No-op, return unchanged arrays
2. **Charge array is None:** Works exactly as before, no charge handling
3. **All charges are zero in colliding pairs:** Optimized early return, no charge operations
4. **Mixed positive/negative charges:** Algebraic sum (can result in neutral particle)
5. **1D distribution:** Charge handling works regardless of distribution dimensionality

## Notes

- The charge array is already sorted in `_enforce_increasing_bins()` along with distribution and concentration, so no changes needed there
- The `coagulation_strategy_abc.py` calls `particle.collide_pairs(loss_gain_indices)` - no changes needed since the public signature is unchanged
- Follows Google-style docstring convention per `docs/Agent/code_style.md`
- Type hints use `Optional[NDArray]` pattern per repository conventions

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-02 | Initial feature documentation | ADW Workflow |
| 2025-12-02 | Implementation complete | ADW Workflow |
