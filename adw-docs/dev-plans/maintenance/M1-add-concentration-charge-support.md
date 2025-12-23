# Maintenance: Add Charge Support to add_concentration

**ID:** M1
**Priority:** P2
**Status:** Shipped
**Last Updated:** 2025-12-23

## Priority Justification

Users simulating ion-aerosol interactions need to periodically refresh ion
concentrations during coagulation simulations. Currently, `add_concentration()`
does not support specifying charge for newly added particles, forcing
workarounds or preventing accurate charged particle injection.

The `collide_pairs()` method already handles charge conservation, but
`add_concentration()` lacks parity, creating an inconsistent API for particle
operations.

## Scope

**Affected Modules/Components:**
- `particula.particles.representation` - Main `ParticleRepresentation` class
- `particula.particles.distribution_strategies.base` - Abstract base class
- `particula.particles.distribution_strategies.particle_resolved_speciated_mass` - Primary target
- `particula.particles.distribution_strategies.mass_based_moving_bin` - Bin-based strategy
- `particula.particles.distribution_strategies.radii_based_moving_bin` - Bin-based strategy
- `particula.particles.distribution_strategies.speciated_mass_moving_bin` - Bin-based strategy

**Affected Files/Directories:**
- `particula/particles/representation.py`
- `particula/particles/distribution_strategies/base.py`
- `particula/particles/distribution_strategies/particle_resolved_speciated_mass.py`
- `particula/particles/distribution_strategies/mass_based_moving_bin.py`
- `particula/particles/distribution_strategies/radii_based_moving_bin.py`
- `particula/particles/distribution_strategies/speciated_mass_moving_bin.py`
- `particula/particles/distribution_strategies/tests/`

## User Story

**As a** researcher simulating ion-aerosol coagulation
**I want** to add charged particles (e.g., 5 nm ions) to an existing aerosol population
**So that** I can refresh ion concentrations periodically during simulations of
ion/small-particle interactions with larger aerosol particles (e.g., 20 nm)

**Example Scenario:**
```python
# Simulation loop for ion-aerosol coagulation
for step in range(n_steps):
    # Coagulation removes ions as they attach to aerosol
    aerosol = coagulation.execute(aerosol, time_step=dt)
    
    # Refresh ion concentration every N steps
    if step % refresh_interval == 0:
        aerosol.particles.add_concentration(
            added_concentration=ion_concentration,
            added_distribution=ion_mass_distribution,
            added_charge=ion_charges,  # NEW: specify charge for ions
        )
```

## Guidelines

### Requirements

1. Add `added_charge` parameter to `ParticleRepresentation.add_concentration()`
2. Update strategy `add_concentration()` signatures to accept optional charge
3. Implement charge handling appropriate to each strategy type:
   - **ParticleResolved**: Append/assign charge values for new particles
   - **Bin-based**: Update charge via concentration-weighted average
4. Default to charge=0 for new particles when `added_charge=None`
5. Maintain backward compatibility (existing code without charge arg still works)

### Standards

- Follow existing pattern from `collide_pairs()` for charge handling
- Use keyword-only argument for `added_charge` to improve API clarity
- Update docstrings to document behavior differences between strategies
- Maintain 80-character line length and Google-style docstrings

### Constraints

- Must not break existing `add_concentration()` calls (backward compatible)
- Charge array shapes must match concentration array shapes
- For bin-based strategies, charge update is concentration-weighted average

## Phases

### Phase 1: ParticleResolved Strategy Charge Support (M1-P1)

**Scope:** Add charge support to `ParticleResolvedSpeciatedMass.add_concentration()`
and `ParticleRepresentation.add_concentration()`.

**Size:** S (~50-70 LOC including tests)

**Tasks:**
- [x] Update `DistributionStrategy.add_concentration()` abstract signature to
      include `added_charge: Optional[NDArray[np.float64]] = None`
- [x] Update return type to include optional charge array
- [x] Implement charge handling in `ParticleResolvedSpeciatedMass.add_concentration()`:
  - Append charge values for new particles
  - Fill empty bins with provided charge or default to 0
- [x] Update `ParticleRepresentation.add_concentration()` to:
  - Accept `*, added_charge: Optional[NDArray[np.float64]] = None` (keyword-only)
  - Pass charge to strategy and update `self.charge`
- [x] Add tests for particle-resolved charge addition scenarios
- [x] Update docstrings documenting ParticleResolved behavior

**Acceptance Criteria:**
- [x] New ions can be added with specified charges
- [x] Missing charge defaults to 0 for new particles
- [x] Existing calls without `added_charge` continue to work
- [x] Tests cover append, fill-empty, and partial-fill scenarios with charge

### Phase 2: Bin-Based Strategy Charge Support (M1-P2)

**Scope:** Add charge support to bin-based strategies with concentration-weighted
averaging.

**Size:** S (~50-70 LOC including tests)

**Tasks:**
- [x] Implement charge handling in `MassBasedMovingBin.add_concentration()`:
  - Compute concentration-weighted average charge when adding to bins
  - Default to existing charge when `added_charge=None`
- [x] Implement same pattern in `RadiiBasedMovingBin.add_concentration()`
- [x] Implement same pattern in `SpeciatedMassMovingBin.add_concentration()`
- [x] Add tests for bin-based charge averaging scenarios
- [x] Update docstrings documenting bin-based behavior differences

**Acceptance Criteria:**
- [x] Adding concentration updates charge via weighted average
- [x] Missing charge preserves existing bin charges
- [x] Docstrings clearly explain bin-based vs particle-resolved behavior

### Phase 3: Documentation Update (M1-P3)

**Scope:** Update development documentation with completion notes.

**Size:** XS

**Tasks:**
- [x] Update this maintenance document status to Shipped
- [x] Add completion notes and lessons learned
- [x] Update README.md in development_plans if needed

## Technical Approach

### API Changes

**`ParticleRepresentation.add_concentration()` (updated signature):**
```python
def add_concentration(
    self,
    added_concentration: NDArray[np.float64],
    added_distribution: Optional[NDArray[np.float64]] = None,
    *,
    added_charge: Optional[NDArray[np.float64]] = None,
) -> None:
    """Add concentration to the particle distribution.

    Arguments:
        added_concentration: The concentration to be added per bin (1/m^3).
        added_distribution: Optional distribution array to merge into the
            existing distribution. If None, the current distribution is reused.
        added_charge: Optional charge array for newly added particles.
            For ParticleResolved strategies, these are assigned to new particles.
            For bin-based strategies, charge is updated via concentration-weighted
            average. If None, defaults to 0 for new particles or preserves
            existing charge for bin-based strategies.
    """
```

**`DistributionStrategy.add_concentration()` (updated signature):**
```python
@abstractmethod
def add_concentration(
    self,
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    added_distribution: NDArray[np.float64],
    added_concentration: NDArray[np.float64],
    charge: Optional[NDArray[np.float64]] = None,
    added_charge: Optional[NDArray[np.float64]] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]:
    """Add concentration to the distribution of particles.

    Arguments:
        distribution: The distribution of particle sizes or masses.
        concentration: The concentration of each particle size or mass.
        added_distribution: The distribution to be added.
        added_concentration: The concentration to be added.
        charge: Current charge array (optional).
        added_charge: Charge for added particles (optional).

    Returns:
        Tuple of (updated distribution, updated concentration, updated charge).
    """
```

### Behavior by Strategy Type

| Strategy | Charge Behavior | Default (added_charge=None) |
|----------|----------------|----------------------------|
| ParticleResolvedSpeciatedMass | Assign/append charge values | New particles get charge=0 |
| MassBasedMovingBin | Concentration-weighted average | Preserve existing charge |
| RadiiBasedMovingBin | Concentration-weighted average | Preserve existing charge |
| SpeciatedMassMovingBin | Concentration-weighted average | Preserve existing charge |

### Concentration-Weighted Average Formula (Bin-Based)

For bin `i` when adding concentration:
```
new_charge[i] = (old_concentration[i] * old_charge[i] + added_concentration[i] * added_charge[i]) 
                / (old_concentration[i] + added_concentration[i])
```

## Success Criteria

- [ ] `add_concentration()` accepts optional `added_charge` parameter
- [ ] ParticleResolved strategy assigns charge to new particles
- [ ] Bin-based strategies update charge via concentration-weighted average
- [ ] Default behavior (no charge arg) maintains backward compatibility
- [ ] Docstrings clearly document strategy-specific behavior
- [ ] All tests passing with >95% coverage on new code
- [ ] Code review approved

## Dependencies

### Upstream Dependencies
- None (self-contained change)

### Downstream Dependencies
- Coagulation simulations using ion injection
- Any future code relying on `add_concentration()` with charged particles

## Testing Strategy

### Unit Tests (Phase 1 - ParticleResolved)
- [ ] Add particles with specified charges (append case)
- [ ] Add particles with charges filling empty bins
- [ ] Add particles with charges (partial fill + append)
- [ ] Add particles without charge arg (defaults to 0)
- [ ] Shape mismatch between added_concentration and added_charge raises error

### Unit Tests (Phase 2 - Bin-Based)
- [ ] Add concentration with charge updates weighted average
- [ ] Add concentration without charge preserves existing charge
- [ ] Weighted average calculation correctness
- [ ] Edge case: adding to bin with zero concentration

### Integration Tests
- [ ] Full workflow: create representation, add charged particles, verify state
- [ ] Coagulation + ion refresh simulation pattern works correctly

## Context

### Current State
- `add_concentration()` ignores charge entirely
- `collide_pairs()` handles charge correctly (sets precedent)
- Users cannot inject charged particles without manual array manipulation

### Desired State
- `add_concentration()` has parity with `collide_pairs()` for charge handling
- Users can refresh ion concentrations with specified charges in simulations
- Clear documentation explains behavior differences between strategies

### Impact
- **User Impact:** Enables ion-aerosol coagulation simulations with periodic ion refresh
- **Developer Impact:** Consistent charge handling across particle operations
- **System Impact:** Minor API addition, fully backward compatible

## Completion Notes

### Implementation Summary

- **Phase 1 (#78):** Added keyword-only `added_charge` handling in
  `ParticleRepresentation` and particle-resolved strategy, including tests for
  append/fill paths.
- **Phase 2 (#79):** Implemented concentration-weighted charge updates across
  bin-based strategies with documentation and coverage.

### Lessons Learned

- Align charge array shape validation early to avoid downstream errors.
- Maintain parity between particle-resolved and bin-based docstrings to reduce
  support questions.

### Actual vs Planned

- Matched plan; all phases shipped as scheduled with backward compatibility
  confirmed.

## References

- Related Architecture: Strategy pattern in `distribution_strategies/`
- Related Code: `collide_pairs()` charge handling pattern
- Related Issues: None yet (to be created)

## Notes

- The keyword-only `added_charge` parameter (using `*`) ensures clear API usage
  and prevents positional argument confusion.
- Bin-based strategies use concentration-weighted averaging because bins represent
  populations, not individual particles. Adding concentration to a bin is like
  mixing two populations with potentially different average charges.
- ParticleResolved strategies assign charges directly because each array element
  represents an individual particle.

---

**Instructions for ADW Workflow:**
When generating issues from this maintenance plan:
1. Create Phase 1 issue first (ParticleResolved + representation changes)
2. Create Phase 2 issue after Phase 1 merges (bin-based strategies)
3. Reference this document in issue descriptions
4. Apply labels: `maintenance`, `priority:P2`, `particles`, `enhancement`

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-19 | Initial maintenance plan created | ADW Workflow |
| 2025-12-23 | Feature shipped - all phases complete | ADW Workflow |
