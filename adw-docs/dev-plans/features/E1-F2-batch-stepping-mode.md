# Feature E1-F2: Batch-Wise Stepping Mode

**Status:** Not Started
**Priority:** P2
**Assignees:** TBD
**Labels:** feature, dynamics, condensation, staggered-stepping, batch
**Milestone:** v0.3.x
**Size:** S (~100 LOC core + tests)

**Start Date:** TBD
**Target Date:** TBD
**Created:** 2025-12-23
**Updated:** 2025-12-23

**Parent Epic:** [E1: Staggered ODE Stepping][epic-e1]
**Related Issues:** TBD
**Related PRs:** TBD

---

## Overview

Extend the staggered stepping framework with batch-wise Gauss-Seidel stepping
that processes particle batches sequentially, updating gas concentration after
each batch completes. This mode provides a middle ground between fully
staggered (per-particle updates) and simultaneous (all-at-once) approaches.

### Problem Statement

While the core staggered stepping (E1-F1) provides per-particle theta values,
some simulations benefit from:

- Processing particles in groups rather than individually (performance)
- Gauss-Seidel-style batch updates where gas is updated between batches
- Combining batching with theta modes (e.g., random theta within batches)

### Value Proposition

- **Performance**: Batch processing allows vectorized operations within batches
- **Flexibility**: Configurable batch count (1 to n_particles)
- **Numerical Properties**: Gauss-Seidel iteration improves convergence
- **Composability**: Batching combines with any theta mode

## Scope

### In Scope

- Batch-wise Gauss-Seidel stepping within `CondensationIsothermalStaggered`
- Sequential batch processing with gas updates between batches
- Combining batching with theta modes
- Batch size validation and edge case handling
- Unit tests for batch behavior

### Out of Scope

- Core class skeleton (E1-F1)
- Builder and factory integration (E1-F3)
- Mass conservation validation tests (E1-F4)
- Performance benchmarks comparing batch sizes (E1-F5)

## Dependencies

### Upstream

- **E1-F1**: Core Staggered Stepping Logic (must be complete)
  - Provides `CondensationIsothermalStaggered` class
  - Provides `_make_batches()` method foundation

### Downstream

- E1-F3: Builder and Factory Integration
- E1-F4: Mass Conservation Validation
- E1-F5: Stability and Performance Benchmarks

## Phase Checklist

- [ ] **E1-F2-P1:** Implement batch-wise Gauss-Seidel stepping with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Process batches sequentially within each pass
  - Update gas concentration after each batch completes
  - Support combining batching with theta modes (e.g., random theta within
    batches)
  - Include tests for batch ordering effects and gas update timing

- [ ] **E1-F2-P2:** Add batch size validation and edge cases with tests
  - Issue: TBD | Size: XS | Status: Not Started
  - Handle `num_batches > n_particles` gracefully (reduce to n_particles)
  - Handle `num_batches = 1` (equivalent to no batching)
  - Validate `num_batches >= 1`
  - Include edge case tests for all boundary conditions

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds at 80% or higher.
- **Self-Contained Tests**: Ship `*_test.py` suites that prove batch stepping
  works correctly.
- **Test-First Completion**: Tests must exist and pass before finishing each
  phase.
- **80%+ Coverage**: Maintain at least 80% coverage for touched code.

## Testing Strategy

### Unit Tests

Location: `particula/dynamics/condensation/tests/condensation_strategies_test.py`

**Test Cases:**

- [ ] `num_batches=1` processes all particles in single batch
- [ ] `num_batches=n_particles` processes each particle individually
- [ ] `num_batches > n_particles` reduces to `n_particles` batches
- [ ] `num_batches=0` or negative raises ValueError
- [ ] Gas concentration updates between batches (not just at end)
- [ ] Batch ordering affects final state (non-commutative)

### Integration Tests

Location: `particula/dynamics/condensation/tests/condensation_strategies_test.py`

**Test Cases:**

- [ ] Batch stepping with `theta_mode="half"` produces valid output
- [ ] Batch stepping with `theta_mode="random"` produces valid output
- [ ] Different `num_batches` values produce different (but valid) results
- [ ] Shuffling changes batch composition across steps

### Behavior Tests

- [ ] Gauss-Seidel property: earlier batches affect later batches within step
- [ ] Two passes both respect batch structure
- [ ] Total mass change is independent of batch count (within tolerance)

## Technical Approach

### Algorithm

```python
def step(self, particle, gas_species, temperature, pressure, time_step):
    theta = self._get_theta_values(n_particles)
    batches = self._make_batches(particle_indices)

    gas_concentration = gas_species.concentration.copy()

    # Pass 1: fractional step with theta
    for batch in batches:
        batch_dm_total = 0
        for idx in batch:
            dt_local = theta[idx] * time_step
            dm = self._condense_single(particle[idx], gas_concentration, dt_local)
            particle.mass[idx] += dm
            batch_dm_total += dm
        # Gauss-Seidel: update gas after each batch
        gas_concentration -= batch_dm_total / air_volume

    # Pass 2: remaining step with (1 - theta)
    for batch in batches:
        batch_dm_total = 0
        for idx in batch:
            dt_local = (1 - theta[idx]) * time_step
            dm = self._condense_single(particle[idx], gas_concentration, dt_local)
            particle.mass[idx] += dm
            batch_dm_total += dm
        gas_concentration -= batch_dm_total / air_volume

    return particle, updated_gas_species
```

### Batch Behavior Matrix

| num_batches | Behavior | Use Case |
|-------------|----------|----------|
| 1 | All particles in one batch, single gas update | Fast, less accurate |
| n_particles | Each particle is own batch, per-particle gas update | Most accurate, slower |
| 10 | Groups of ~n/10, balanced tradeoff | Typical simulation |
| > n_particles | Clipped to n_particles | Edge case handling |

### Edge Case Handling

```python
def _validate_num_batches(self, num_batches: int, n_particles: int) -> int:
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    # Silently clip to n_particles (can't have more batches than particles)
    return min(num_batches, n_particles)
```

## Success Criteria

- [ ] Batch-wise stepping processes batches sequentially with gas updates
- [ ] `num_batches` validation handles all edge cases correctly
- [ ] Combining batching with theta modes works correctly
- [ ] Gauss-Seidel property verified (batch order matters)
- [ ] All unit and integration tests pass
- [ ] Code coverage >= 80% for new code
- [ ] Code review approved

## Usage Example

```python
from particula.dynamics.condensation import CondensationIsothermalStaggered

# Create staggered strategy with batch stepping
strategy = CondensationIsothermalStaggered(
    molar_mass=0.018,
    theta_mode="random",
    num_batches=10,  # Process in 10 batches
    shuffle_each_step=True,  # Randomize batch composition
)

# Each step processes 10 batches with gas updates between them
particle, gas = strategy.step(
    particle=particle,
    gas_species=gas_species,
    temperature=298.0,
    pressure=101325.0,
    time_step=1.0,
)
```

## Change Log

| Date       | Change                                | Author       |
|------------|---------------------------------------|--------------|
| 2025-12-23 | Initial feature documentation created | ADW Workflow |

[epic-e1]: ../epics/E1-staggered-condensation-stepping.md
