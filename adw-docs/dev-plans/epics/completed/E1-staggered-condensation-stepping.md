# Epic E1: Staggered ODE Stepping for Particle-Resolved Condensation

**Status**: Completed
**Priority**: P2
**Owners**: @Gorkowski
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-01-03
**Size**: Medium (6-8 features, ~20 phases)

## Vision

Improve the **stability**, **efficiency**, and **physical realism** of particle-resolved condensation time stepping in particula. The current `CondensationIsothermal.step()` updates all particles simultaneously against a fixed gas concentration, which can cause numerical instability and unrealistic vapor competition at large time steps.

This epic introduces a **staggered ODE stepping framework** that:
- Preserves **mass conservation** (total gas + particle mass unchanged)
- Handles **Kelvin effect** and competitive vapor uptake naturally
- Reduces **numerical noise** from simultaneous updates
- Avoids expensive fully-implicit or equilibrium solvers

The framework provides three configurable stepping modes:
1. **Fixed half-step**: Deterministic, mildly staggered (θ = 0.5 for all particles)
2. **Randomized fractional step**: Each particle gets unique θ ∈ [0,1], reducing systematic bias
3. **Batch-wise stepping**: Gauss-Seidel-style updates with configurable batch count

## Scope

### In Scope

- New `CondensationIsothermalStaggered` strategy class extending `CondensationStrategy`
- Three stepping modes: `"half"`, `"random"`, `"batch"`
- Support for `ParticleResolvedSpeciatedMass` distribution strategy only (initial scope)
- Builder and factory integration for the new strategy
- Comprehensive tests for mass conservation, stability, and performance
- Documentation and usage examples

### Out of Scope

- Support for discrete bin distributions (`MassBasedMovingBin`, etc.) — future epic
- Support for continuous PDF distributions — future epic
- New runnable wrapper (`StaggeredMassCondensation`) — may refactor after usage experience
- Latent heat / non-isothermal condensation — separate feature
- JAX/GPU acceleration — separate optimization epic

## Dependencies

- None blocking. The condensation module (`particula/dynamics/condensation/`) is stable.
- Requires `ParticleResolvedSpeciatedMass` distribution strategy (already implemented)

## Literature References

This approach draws on established numerical methods:

1. **Operator Splitting**: LeVeque (2002) — *Finite Volume Methods for Hyperbolic Problems*
2. **Symplectic Integration**: Hairer, Lubich, & Wanner (2006) — *Geometric Numerical Integration*
3. **Mass-Conserving Condensation**: Jacobson (1997, 1999) — Analytical Predictor of Condensation (APC)
4. **Particle-Resolved Modeling**: Riemer et al. (2009) — PartMC-MOSAIC
5. **Stochastic Particle Methods**: Kumar et al. (2014) — Stochastic particle-resolved aerosol modeling

## Phase Checklist

### Feature F1: Core Staggered Stepping Logic

- [x] **E1-F1-P1**: Create `CondensationIsothermalStaggered` class skeleton with tests
  - File: `particula/dynamics/condensation/condensation_strategies.py`
  - Add new class inheriting from `CondensationStrategy`
  - Add `theta_mode` parameter: `"half"`, `"random"`, `"batch"`
  - Add `num_batches` parameter (default: 1)
  - Add `shuffle_each_step` parameter (default: True for random mode)
  - Include unit tests for class instantiation and parameter validation

- [x] **E1-F1-P2**: Implement `_get_theta_values()` helper method with tests
  - Generate θ array based on mode:
    - `"half"`: `np.full(n_particles, 0.5)`
    - `"random"`: `np.random.uniform(0, 1, n_particles)`
    - `"batch"`: `np.ones(n_particles)` (batching handled separately)
  - Include unit tests for each mode

- [x] **E1-F1-P3**: Implement `_make_batches()` helper method with tests
  - Shuffle particle indices if `shuffle_each_step=True`
  - Divide into `num_batches` groups
  - Return list of index arrays
  - Include unit tests for batch creation and shuffling

- [x] **E1-F1-P4**: Implement two-pass staggered `step()` method with tests
  - Pass 1: Each particle grows for `θ_p × Δt`, update gas cumulatively
  - Pass 2: Each particle grows for `(1 - θ_p) × Δt` with updated gas
  - Preserve existing API: `step(particle, gas_species, temperature, pressure, time_step)`
  - Include integration tests verifying mass conservation

### Feature F2: Batch-Wise Stepping Mode

- [x] **E1-F2-P1**: Implement batch-wise Gauss-Seidel stepping with tests
  - Process batches sequentially within each pass
  - Update gas concentration after each batch completes
  - Support combining batching with θ modes (e.g., random θ within batches)
  - Include tests for batch ordering effects

- [x] **E1-F2-P2**: Add batch size validation and edge cases with tests
  - Handle `num_batches > n_particles` gracefully
  - Handle `num_batches = 1` (equivalent to no batching)
  - Validate `num_batches >= 1`
  - Include edge case tests

### Feature F3: Builder and Factory Integration

- [x] **E1-F3-P1**: Create `CondensationIsothermalStaggeredBuilder` with tests
  - File: `particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py`
  - Methods: `set_theta_mode()`, `set_num_batches()`, `set_shuffle_each_step()`
  - Inherit common methods from mixin
  - Include builder tests

- [x] **E1-F3-P2**: Register in `CondensationFactory` with tests
  - Add `"isothermal_staggered"` strategy type
  - Wire up factory parameters
  - Include factory tests

- [x] **E1-F3-P3**: Export from `particula.dynamics` namespace with tests
  - Update `particula/dynamics/__init__.py`
  - Update `particula/dynamics/condensation/__init__.py`
  - Include import tests

### Feature F4: Mass Conservation Validation

- [x] **E1-F4-P1**: Create mass conservation test harness with comprehensive tests
  - File: `particula/dynamics/condensation/tests/staggered_mass_conservation_test.py`
  - Test all three modes: half, random, batch
  - Verify `|total_initial - total_final| < tolerance` (1e-12 relative)
  - Test with varying particle counts (100, 1000, 10000)

- [x] **E1-F4-P2**: Add Kelvin effect stress tests
  - Test small particles (< 10 nm) with high Kelvin curvature
  - Verify evaporation occurs correctly under vapor competition
  - Test supersaturation and subsaturation scenarios

### Feature F5: Stability and Performance Benchmarks

- [x] **E1-F5-P1**: Create stability benchmark tests
  - File: `particula/dynamics/condensation/tests/staggered_stability_test.py`
  - Compare variance in particle size distribution vs simultaneous update
  - Test large time steps (1s, 10s, 100s) where simultaneous fails
  - Document stability improvement metrics

- [x] **E1-F5-P2**: Create performance benchmark tests
  - Compare runtime: simultaneous vs staggered modes
  - Test scaling with particle count (1k, 10k, 100k)
  - Document performance characteristics
  - Mark as `@pytest.mark.slow` for CI

### Feature F6: Documentation and Examples

- [x] **E1-F6-P1**: Add docstrings and inline documentation with validation
  - Google-style docstrings for all new public methods
  - Add literature citations in module docstring
  - Run docstring linter to validate

- [x] **E1-F6-P2**: Create usage example notebook
  - File: `docs/Examples/Dynamics/staggered_condensation_example.ipynb`
  - Demonstrate all three modes
  - Show mass conservation verification
  - Compare stability with simultaneous stepping

- [x] **E1-F6-P3**: Update development documentation
  - Update `adw-docs/dev-plans/README.md` with epic status
  - Create `adw-docs/dev-plans/epics/index.md` if needed
  - Add completion notes and lessons learned

## Completion Notes

Completion Date: 2026-01-03  
Summary: Completed 6/6 features and 16/16 phases.

### Implementation Summary
- Delivered staggered condensation strategy with half, random, and batch theta modes.
- Integrated builder and factory pathways with exports for `CondensationIsothermalStaggered`.
- Added mass conservation validation across modes and particle counts with regression thresholds.
- Landed stability/performance benchmarks and marked slow paths for CI hygiene.
- Completed docstrings with citations plus usage tutorial notebook covering all modes.
- Updated development documentation and indices to reflect completion.

### Key Decisions
- Kept theta modes scoped to half/random/batch with shuffle defaults to limit bias.
- Retained two-pass update with cumulative gas updates to prioritize mass conservation.
- Marked performance/stability suites as slow to avoid CI runtime regression.

### Lessons Learned
- Batch ordering and shuffling materially affect variance; deterministic seeds aid comparisons.
- Mass conservation tolerances are sensitive to particle counts; regression tests guard drift.
- Documentation plus notebook examples reduce onboarding time for staggered stepping.

### Actual vs Planned
- All planned features shipped; no scope removed or deferred.
- Performance and stability coverage delivered as separate slow suites per plan.
- Documentation and tutorial assets completed alongside code/tests.

### Future Work
- Extend staggered stepping to discrete-bin and continuous-PDF distributions.
- Explore GPU/JAX acceleration and adaptive batching heuristics after usage feedback.

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds as shipped (80%).
- **Self-Contained Tests**: Attach `*_test.py` files that prove each phase is complete.
- **Test-First Completion**: Write and pass tests before declaring phases ready for review.
- **80%+ Coverage**: Every phase must ship tests that maintain at least 80% coverage.
- **Mass Conservation**: All modes must preserve total mass to within 1e-12 relative tolerance.

## Testing Strategy

### Unit Tests
- `particula/dynamics/condensation/tests/condensation_strategies_test.py` — extend with staggered class tests
- `particula/dynamics/condensation/condensation_builder/tests/condensation_isothermal_staggered_builder_test.py` — new file
- `particula/dynamics/condensation/tests/condensation_factories_test.py` — extend with factory tests

### Integration Tests
- `particula/dynamics/condensation/tests/staggered_mass_conservation_test.py` — new file
- `particula/dynamics/condensation/tests/staggered_stability_test.py` — new file
- `particula/integration_tests/condensation_particle_resolved_test.py` — extend with staggered mode tests

### Performance Tests (marked slow)
- `particula/dynamics/condensation/tests/staggered_performance_test.py` — new file, `@pytest.mark.slow`

## Success Metrics

1. **Mass Conservation**: All modes preserve total (gas + particle) mass to < 1e-12 relative error
2. **Stability**: Staggered modes allow 10x larger time steps than simultaneous before instability
3. **Performance**: < 2x runtime overhead compared to simultaneous stepping for typical use cases
4. **Coverage**: Maintain 80%+ test coverage for new code

## Pseudocode Reference

```python
# Core stepping logic (simplified)
theta = get_theta(mode="random" | "half" | "batch")  # length N
batches = make_batches(particles, num_batches)

for pass_num in [1, 2]:
    for batch in batches:
        dm_total = 0
        for p in batch:
            t_frac = theta[p.idx] if pass_num == 1 else 1 - theta[p.idx]
            dt_local = t_frac * dt
            dm = condense(p, C_gas, dt_local)
            p.mass += dm
            dm_total += dm
        C_gas -= dm_total / V_air
```

## Additional Notes

- After gaining usage experience, we may refactor to a new `StaggeredMassCondensation` runnable for better composability
- Future epics may extend support to discrete bin and continuous PDF distributions
- Consider JAX/NumPy performance comparison in a future optimization epic

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-22 | Initial epic creation | ADW |
| 2025-12-29 | Feature plans reviewed and expanded with: algorithm details, correct file paths, fixture code, risks sections, dependency clarifications | ADW |
