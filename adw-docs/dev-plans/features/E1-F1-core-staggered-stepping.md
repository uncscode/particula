# Feature E1-F1: Core Staggered Stepping Logic

**Status:** Not Started
**Priority:** P2
**Assignees:** TBD
**Labels:** feature, dynamics, condensation, staggered-stepping
**Milestone:** v0.3.x
**Size:** M (~200 LOC core + tests)

**Start Date:** TBD
**Target Date:** TBD
**Created:** 2025-12-23
**Updated:** 2025-12-23

**Parent Epic:** [E1: Staggered ODE Stepping][epic-e1]
**Related Issues:** TBD
**Related PRs:** TBD

---

## Overview

Implement the core `CondensationIsothermalStaggered` strategy class that provides
staggered ODE stepping for particle-resolved condensation. This feature
introduces the fundamental two-pass stepping algorithm with configurable theta
modes (`"half"`, `"random"`, `"batch"`) to improve numerical stability and mass
conservation.

### Problem Statement

The current `CondensationIsothermal.step()` updates all particles simultaneously
against a fixed gas concentration. At large time steps, this causes:

- Numerical instability from competing vapor uptake
- Unrealistic mass transfer when multiple particles deplete the same vapor
- Accumulated error in total mass conservation

### Value Proposition

- **Mass Conservation**: Two-pass algorithm preserves total (gas + particle) mass
- **Stability**: Staggered updates reduce numerical noise from simultaneous
  growth
- **Flexibility**: Three theta modes support different use cases (deterministic
  vs stochastic vs batch)
- **API Compatibility**: Preserves existing `step()` signature for drop-in
  replacement

## Scope

### In Scope

- `CondensationIsothermalStaggered` class extending `CondensationStrategy`
- Three theta modes: `"half"`, `"random"`, `"batch"`
- `_get_theta_values()` helper method for theta array generation
- `_make_batches()` helper method for particle index batching
- Two-pass `step()` implementation with cumulative gas updates
- Unit tests for all new methods and modes

### Out of Scope

- Builder and factory integration (E1-F3)
- Batch-wise Gauss-Seidel stepping details (E1-F2)
- Mass conservation validation tests (E1-F4)
- Performance benchmarks (E1-F5)
- Documentation and examples (E1-F6)

## Dependencies

### Upstream

- None blocking. Condensation module is stable.
- Requires `ParticleResolvedSpeciatedMass` distribution strategy (exists)

### Downstream

- E1-F2: Batch-Wise Stepping Mode (extends batch logic)
- E1-F3: Builder and Factory Integration (wraps this class)
- E1-F4: Mass Conservation Validation (tests this class)

## Phase Checklist

- [ ] **E1-F1-P1:** Create `CondensationIsothermalStaggered` class skeleton with
      tests
  - Issue: TBD | Size: S | Status: Not Started
  - File: `particula/dynamics/condensation/condensation_strategies.py`
  - Add new class inheriting from `CondensationStrategy`
  - Add `theta_mode` parameter: `"half"`, `"random"`, `"batch"`
  - Add `num_batches` parameter (default: 1)
  - Add `shuffle_each_step` parameter (default: True for random mode)
  - Include unit tests for class instantiation and parameter validation

- [ ] **E1-F1-P2:** Implement `_get_theta_values()` helper method with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Generate theta array based on mode:
    - `"half"`: `np.full(n_particles, 0.5)`
    - `"random"`: `np.random.uniform(0, 1, n_particles)`
    - `"batch"`: `np.ones(n_particles)` (batching handled separately)
  - Include unit tests for each mode verifying shape and value ranges

- [ ] **E1-F1-P3:** Implement `_make_batches()` helper method with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Shuffle particle indices if `shuffle_each_step=True`
  - Divide indices into `num_batches` groups
  - Return list of index arrays
  - Include unit tests for batch creation, shuffling, and edge cases

- [ ] **E1-F1-P4:** Implement two-pass staggered `step()` method with tests
  - Issue: TBD | Size: M | Status: Not Started
  - Pass 1: Each particle grows for `theta_p * dt`, update gas cumulatively
  - Pass 2: Each particle grows for `(1 - theta_p) * dt` with updated gas
  - Preserve existing API: `step(particle, gas_species, temperature, pressure,
    time_step)`
  - Include integration tests verifying basic mass conservation

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds at 80% or higher.
- **Self-Contained Tests**: Ship `*_test.py` suites that prove the feature works
  end-to-end.
- **Test-First Completion**: Tests must exist and pass before finishing each
  phase.
- **80%+ Coverage**: Maintain at least 80% coverage for touched code.

## Testing Strategy

### Unit Tests

Location: `particula/dynamics/condensation/tests/condensation_strategies_test.py`

**Test Cases:**

- [ ] Class instantiation with valid parameters succeeds
- [ ] Invalid `theta_mode` raises ValueError
- [ ] Invalid `num_batches` (< 1) raises ValueError
- [ ] `_get_theta_values("half")` returns array of 0.5 values
- [ ] `_get_theta_values("random")` returns values in [0, 1]
- [ ] `_get_theta_values("batch")` returns array of 1.0 values
- [ ] `_make_batches()` divides indices into correct number of batches
- [ ] `_make_batches()` shuffles indices when `shuffle_each_step=True`
- [ ] `_make_batches()` preserves indices when `shuffle_each_step=False`

### Integration Tests

Location: `particula/dynamics/condensation/tests/condensation_strategies_test.py`

**Test Cases:**

- [ ] `step()` with `theta_mode="half"` produces valid output
- [ ] `step()` with `theta_mode="random"` produces valid output
- [ ] `step()` API matches existing `CondensationIsothermal.step()` signature
- [ ] Output particle masses are non-negative
- [ ] Gas concentration remains non-negative

## Technical Approach

### Architecture

```
CondensationStrategy (ABC)
    └── CondensationIsothermal (existing)
    └── CondensationIsothermalStaggered (new)
            ├── theta_mode: str
            ├── num_batches: int
            ├── shuffle_each_step: bool
            ├── _get_theta_values() -> NDArray
            ├── _make_batches() -> list[NDArray]
            └── step() -> tuple[Particle, GasSpecies]
```

### Algorithm (Simplified)

```python
def step(self, particle, gas_species, temperature, pressure, time_step):
    theta = self._get_theta_values(n_particles)
    batches = self._make_batches(particle_indices)

    gas_concentration = gas_species.concentration.copy()

    # Pass 1: fractional step with theta
    for batch in batches:
        for idx in batch:
            dt_local = theta[idx] * time_step
            dm = self._condense_single(particle[idx], gas_concentration, dt_local)
            particle.mass[idx] += dm
            gas_concentration -= dm / air_volume

    # Pass 2: remaining step with (1 - theta)
    for batch in batches:
        for idx in batch:
            dt_local = (1 - theta[idx]) * time_step
            dm = self._condense_single(particle[idx], gas_concentration, dt_local)
            particle.mass[idx] += dm
            gas_concentration -= dm / air_volume

    return particle, updated_gas_species
```

### API Surface

```python
from particula.dynamics.condensation import CondensationIsothermalStaggered

strategy = CondensationIsothermalStaggered(
    molar_mass=0.018,  # kg/mol (water)
    theta_mode="random",  # or "half", "batch"
    num_batches=1,
    shuffle_each_step=True,
)

particle, gas = strategy.step(
    particle=particle,
    gas_species=gas_species,
    temperature=298.0,
    pressure=101325.0,
    time_step=1.0,
)
```

## Success Criteria

- [ ] `CondensationIsothermalStaggered` class instantiates with all theta modes
- [ ] `_get_theta_values()` produces correct arrays for each mode
- [ ] `_make_batches()` correctly divides and optionally shuffles indices
- [ ] `step()` executes two-pass algorithm and returns valid particle/gas
- [ ] All unit and integration tests pass
- [ ] Code coverage >= 80% for new code
- [ ] Code review approved

## Usage Example

```python
import numpy as np
from particula.dynamics.condensation import CondensationIsothermalStaggered

# Create staggered condensation strategy
strategy = CondensationIsothermalStaggered(
    molar_mass=0.018,  # kg/mol for water
    theta_mode="random",
    num_batches=1,
    shuffle_each_step=True,
)

# Step the condensation process
particle_updated, gas_updated = strategy.step(
    particle=particle,
    gas_species=gas_species,
    temperature=298.0,
    pressure=101325.0,
    time_step=0.1,
)
```

## Change Log

| Date       | Change                                | Author       |
|------------|---------------------------------------|--------------|
| 2025-12-23 | Initial feature documentation created | ADW Workflow |

[epic-e1]: ../epics/E1-staggered-condensation-stepping.md
